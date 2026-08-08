# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Install required dependencies
# MAGIC %uv pip install sentence-transformers psycopg2-binary

# COMMAND ----------

# DBTITLE 1,Restart Python environment
dbutils.library.restartPython()

# COMMAND ----------

