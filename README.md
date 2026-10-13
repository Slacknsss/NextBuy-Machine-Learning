# NextBuy

Epitech Paris, B-DAT-201 (data science). Team of three: Simon Slack, Raphaël Sebag, Raphaël Léger.

Exploratory analysis of the Instacart grocery dataset (orders, products, departments), with a few business questions and two small classification models. Everything is in Jupyter notebooks.

## What's in the repo

| File | Content |
|---|---|
| `notebook.ipynb` | Final notebook, merges everyone's part (questions Q1 to Q8) |
| `notebookSLACK.ipynb` | My notebook: best sellers per department, healthy vs unhealthy, Open Food Facts API, healthy classifier |
| `notebookSebag.ipynb` | Raphaël Sebag: peak days, weekend products, reorder model |
| `notebookleger.ipynb` | Raphaël Léger: data loading, peak hours, first item in cart, cart size |

The dataset is not in the repo (Instacart CSVs, around 13.7M order lines). The notebooks expect them in `./datasets/`.

## The two models

**Healthy or not, from the product name (my part, Q8).** TF-IDF on product names, then Logistic Regression and Random Forest.

The first version scored about 99% accuracy. That was a data leak: the `is_healthy` label was built from a list of keywords, and the model was reading the same product names, so it just learned our keyword list. I relabeled the products with real nutrition values from the Open Food Facts API (per 100 g: under 200 kcal, under 5 g sugar, under 3 g fat, over 5 g protein, healthy if 3 out of 4).

Result on the honest labels: 495 labeled products (256 not healthy, 239 healthy), 99 in the test set.

| Model | Accuracy |
|---|---|
| Always predict "not healthy" | 0.517 (share of the majority class) |
| Logistic Regression | 0.687 |
| Random Forest | 0.677 |

So the model learns something from the name alone, but not a lot. The dataset is small because the Open Food Facts API is rate limited.

**High reorder rate per product (Raphaël Sebag, Q7).** Product-level features (total orders, cart position, unique users, organic) to predict if a product is often reordered. Random Forest gets an AUC of 0.997 on 9,814 test products.

Known issue: `reorder_rate` ends up being the raw reorder count (the division is split over two lines), and the reorder count is roughly total orders minus unique users, two of the features. So that 0.997 is mostly a leak too, not a real prediction.

## Run it

```bash
git clone https://github.com/Slacknsss/NextBuy-Machine-Learning.git
cd NextBuy-Machine-Learning
pip install pandas numpy matplotlib seaborn squarify requests scikit-learn jupyter
jupyter notebook notebook.ipynb
```
