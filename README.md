# Category Prediction
* Multiclass category classification of article using Logistic Regression (schikit-learn)
* Input data
	* `data/article_train_data.tsv` - training data (sample)
	* `data/significant_terms.txt` - significant terms of each categories, it is used for dictionary (sample)
* `train.sh` - train data, dictionary file and serialize model are generated
*  `lib/batch/cross_validation.py` - cross validation script
*  `api.rb` - API entry point, using falcon framework
