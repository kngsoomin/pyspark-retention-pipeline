RAW_DIR       ?= data/raw
OUT_DIR       ?= data/processed
MODEL_DIR     ?= data/models/latest_model
CUTOFF        ?= 2011-06-12
LOOKAHEAD     ?= 180

CV_PARALLELISM ?= 4
CV_METRIC      ?= aucpr           # aucroc | aucpr
THR_METRIC     ?= f1              # f1 | recall | precision

# Plots
SPLIT         ?= val              # train | val | test
THR_PNG       ?= docs/threshold_curve.png
THR_CSV       ?= docs/threshold_curve.csv
FI_PNG        ?= docs/feature_importance.png

PORT          ?= 8000

.PHONY: help init etl train eval metrics coefs \
        serve curl_predict curl_batch \
        plot_threshold plot_importance \
        clean_data clean_model clean_all
help:
	@echo "Targets:"
	@echo "  init              - Install Python deps"
	@echo "  etl               - Build features & splits (train/val/test)"
	@echo "  train             - Train Spark ML pipeline (CV + class weights)"
	@echo "  eval              - Evaluate saved model on test split"
	@echo "  metrics           - Print metrics.json"
	@echo "  coefs             - Show head of coefficients.csv"
	@echo "  serve             - Run FastAPI (uvicorn) on port $(PORT)"
	@echo "  curl_predict      - Sample single prediction against running API"
	@echo "  curl_batch        - Sample batch prediction against running API"
	@echo "  plot_threshold    - Save threshold optimization curve (PNG/CSV)"
	@echo "  plot_importance   - Save signed feature-importance PNG"
	@echo "  clean_data        - Remove processed parquet splits"
	@echo "  clean_model       - Remove saved model artifacts"
	@echo "  clean_all         - Remove processed data and model artifacts"

init:
	pip install -r requirements.txt

etl:
	python pipeline/etl.py \
		--raw_dir $(RAW_DIR) \
		--out_dir $(OUT_DIR) \
		--cutoff $(CUTOFF) \
		--lookahead_days $(LOOKAHEAD)

train:
	python pipeline/train.py \
		--in_dir $(OUT_DIR) \
		--out_dir $(MODEL_DIR) \
		--cv_parallelism $(CV_PARALLELISM) \
		--metric $(CV_METRIC) \
		--thr_metric $(THR_METRIC)

eval:
	python pipeline/eval.py --in_dir $(OUT_DIR) --model_dir $(MODEL_DIR)

metrics:
	@cat $(MODEL_DIR)/metrics.json || echo "metrics.json not found"

coefs:
	@head -n 10 $(MODEL_DIR)/coefficients.csv || echo "coefficients.csv not found"

serve:
	uvicorn serve_api:app --reload --port $(PORT)

curl_predict:
	@curl -s -X POST http://localhost:$(PORT)/predict \
	  -H "Content-Type: application/json" \
	  -d '{"features": {"total_orders": 5, "total_qty": 20, "avg_order_amount": 45.0, "distinct_products": 12, "recent90_orders": 1, "recency_days": 90, "total_amount_log": 6.9, "recent90_amount_log": 4.8}}' \
	| jq .

curl_batch:
	@curl -s -X POST http://localhost:$(PORT)/predict_batch \
	  -H "Content-Type: application/json" \
	  -d '{"items":[{"features":{"total_orders":5,"total_qty":20,"avg_order_amount":45.0,"distinct_products":12,"recent90_orders":1,"recency_days":90,"total_amount_log":6.9,"recent90_amount_log":4.8}},{"features":{"total_orders":2,"total_qty":3,"avg_order_amount":12.0,"distinct_products":2,"recent90_orders":0,"recency_days":170,"total_amount_log":4.5,"recent90_amount_log":0.0}}]}' \
	| jq .

plot_threshold:
	python pipeline/plot_threshold.py \
		--in_dir $(OUT_DIR) \
		--model_dir $(MODEL_DIR) \
		--split $(SPLIT) \
		--out $(THR_PNG) \
		--csv_out $(THR_CSV)

plot_importance:
	python pipeline/plot_feature_importance.py \
		--coef_csv $(MODEL_DIR)/coefficients.csv \
		--out $(FI_PNG)

clean_data:
	rm -rf $(OUT_DIR)

clean_model:
	rm -rf $(MODEL_DIR)

clean_all: clean_data clean_model