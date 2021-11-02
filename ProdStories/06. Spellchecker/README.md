# Word frequencies
Thanks to [this kaggle competition](https://www.kaggle.com/rtatman/english-word-frequency) for english word frequency dataset

# Test
Test data was taken from [here](http://aspell.net/test/cur/)

# Example of evaluation on test data
## Accuracy
### 1st model
* `@1  = 0.52`
* `@3  = 0.65`
* `@5  = 0.66`
* `@10 = 0.67`

### 2nd model (slightly better, ~5%)
* `@1  = 0.56`
* `@3  = 0.67`
* `@5  = 0.69`
* `@10 = 0.72`

## Median position of the correct word in the suggestions
1.00 in both models

that is, almost every time we suggest correct word as the 2nd one.
