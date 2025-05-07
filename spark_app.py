from pyspark.ml.regression import RandomForestRegressor, LinearRegression

import sys
import time
import logging
import json
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, \
    SQLTransformer, RegexTokenizer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('spark_online_retail.log'),
        logging.StreamHandler()
    ]
)
   

def run_spark_job(optimized=False):
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    # Инициализация SparkSession
    spark = SparkSession.builder \
        .appName("OnlineRetailAnalysis") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    
    logger.info("Начало выполнения Spark job")

    # Загрузка данных
    logger.info("Загрузка данных из HDFS")
    df = spark.read.csv("hdfs://namenode:9000/input/Online_Retail_Sample_100k.csv", 
                       header=True, 
                       inferSchema=True)
    logger.info(f"Загружено {df.count()} строк")

    # Предварительная очистка данных
    df = df.filter(
        (df.Quantity > 0) & 
        (df.UnitPrice > 0) & 
        df.Description.isNotNull()
    )

    # Извлечение признаков из даты
    logger.info("Извлечение признаков из даты")
    date_transformer = SQLTransformer(
        statement="""
        SELECT *, 
            dayofweek(InvoiceDate) as Weekday,
            hour(InvoiceDate) as HourOfDay
        FROM __THIS__
        """
    )

    # Токенизация описания товара
    logger.info("Токенизация описания товара")
    tokenizer = RegexTokenizer(
        inputCol="Description",
        outputCol="words",
        pattern="\\W+"
    )

    # Категориальные признаки
    logger.info("Подготовка категориальных признаков")
    indexers = [
        StringIndexer(inputCol="Country", outputCol="CountryIndex"),
        StringIndexer(inputCol="StockCode", outputCol="StockCodeIndex")
    ]

    encoders = [
        OneHotEncoder(inputCol="CountryIndex", outputCol="CountryVec"),
        OneHotEncoder(inputCol="StockCodeIndex", outputCol="StockCodeVec")
    ]

    # Сборка признаков
    assembler = VectorAssembler(
        inputCols=[
            "UnitPrice", "HourOfDay", "Weekday", 
            "CountryVec", "StockCodeVec"
        ],
        outputCol="features"
    )

    # Пайплайн
    pipeline = Pipeline(stages=[
        date_transformer,
        tokenizer,
        *indexers, *encoders,
        assembler
    ])

    # Обработка данных
    model = pipeline.fit(df)
    df_transformed = model.transform(df)

    if optimized:
        logger.info("Применение оптимизаций: repartition + persist")
        df_transformed = df_transformed.repartition(8).persist()
        df_transformed.count()

    # Разделение данных
    logger.info("Разделение данных на train/test")
    train_data, test_data = df_transformed.randomSplit([0.8, 0.2], seed=123)


    logger.info("Обучение модели Random Forest")
    model = RandomForestRegressor(
        featuresCol='features',
        labelCol='Quantity',
        seed=42,
        numTrees=20,     # количество деревьев
        maxDepth=5,      # глубина деревьев
        maxBins=64       # количество бинов для категориальных признаков
    )
    
    # Обучение модели
    trained_model = model.fit(train_data)
    
    # Оценка
    predictions = trained_model.transform(test_data)
    evaluator = RegressionEvaluator(
        labelCol="Quantity",
        predictionCol="prediction",
        metricName="rmse"
    )
    rmse = evaluator.evaluate(predictions)
    
    # Сохранение результатов
    result = {
        "runtime_sec": round(runtime, 2),
        "mem_used": f"{mem_used:.2f} MiB",
        "rmse": rmse,
        "optimized": optimized,
        "model": model_type  # тип модели
    }
    print(json.dumps(result))

    spark.stop()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "lr"],
                        help="Тип модели: 'rf' - Random Forest, 'lr' - Linear Regression")
    parser.add_argument("--optimized", action="store_true", 
                        help="Включить оптимизацию (repartition + persist)")
    args = parser.parse_args()
    
    run_spark_job(model_type=args.model, optimized=args.optimized)