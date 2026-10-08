from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def main():
    # Expected repository layout:
    # dataset/task1/train.csv
    # dataset/task1/test_features.csv
    # outputs/task1_predictions.csv
    repo_root = Path(__file__).resolve().parents[1]
    train_path = repo_root / "dataset" / "task1" / "train.csv"
    test_path = repo_root / "dataset" / "task1" / "test_features.csv"
    output_path = repo_root / "outputs" / "task1_predictions.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target = "prediction"
    id_column = "patient_id"

    X_train = train.drop(columns=[target, id_column])
    y_train = train[target]
    X_test = test.drop(columns=[id_column])

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=15,
                    l2_regularization=2.0,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test).astype(int)

    result = pd.DataFrame(
        {
            id_column: test[id_column],
            target: predictions,
        }
    )

    # Validate the required output contract before saving.
    assert len(result) == len(test)
    assert result[id_column].equals(test[id_column])
    assert list(result.columns) == [id_column, target]
    assert result[target].notna().all()
    assert result[id_column].notna().all()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} predictions to {output_path}")


if __name__ == "__main__":
    main()
