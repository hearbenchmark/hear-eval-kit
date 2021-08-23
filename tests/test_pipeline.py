import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from heareval.tasks.util.luigi import subsample_metadata


@pytest.mark.parametrize(
    ("test_metadata", "max_files", "expected_subsampled_metadata"),
    [
        # Tests - Stratification during standard label distribution
        # (comparatively balanced)
        # Expectation - Size of each label group is 2, and we need to select 3 examples.
        # Examples in l1, l2 and l3 with the smallest split key and subsample key
        # will be selected
        pytest.param(
            [
                ("l1", 1, 1),
                ("l1", 1, 2),
                ("l2", 5, 3),
                ("l2", 1, 4),
                ("l2", 7, 5),
                ("l3", 2, 6),
            ],
            3,
            ([("l1", 1, 1), ("l2", 1, 4), ("l3", 2, 6)]),
            id="1. Test Stratify Key",
        ),
        # Tests - Stratification during Extreme imbalance in the label distribution.
        # Expectation - By stratification the 2 metadata will be picked up from the
        # l2 and since l1 just has 1 sample, that will be picked up to ensure
        # we atleast have one instance of each stratify key
        pytest.param(
            [
                ("l1", 1, 1),
                ("l1", 1, 2),
                ("l2", 5, 3),
                ("l2", 1, 4),
                ("l2", 1, 5),
                ("l2", 1, 6),
                ("l2", 2, 7),
                ("l2", 2, 8),
                ("l2", 2, 9),
            ],
            2,
            [("l1", 1, 1), ("l2", 1, 4)],
            id="2. Test Stratify Key",
        ),
        # Tests - Data points in same with same split key in a label group
        # are selected in chunks
        # Expectation - By stratification we need to select 2 from l1 and 2
        # from l2. From l1, the data point with split key 1 will be selected
        # and from l2, the data point with split key 2 will be selected
        pytest.param(
            [
                ("l1", 1, 1),
                ("l1", 1, 2),
                ("l1", 2, 3),
                ("l1", 2, 4),
                ("l2", 5, 5),
                ("l2", 5, 6),
                ("l2", 2, 7),
                ("l2", 2, 8),
            ],
            4,
            [("l1", 1, 1), ("l1", 1, 2), ("l2", 2, 7), ("l2", 2, 8)],
            id="3. Test Split Key",
        ),
        # Tests - Disabiguity on the basis of subsample key
        # Expectation - By stratification we need to select 3 from l1 and 3 from l2.
        # Here since the split key group in l1 and l2 are of size 2,
        # the third example for each of "l1" and "l2" will be selected on the
        # basis of the subsample key
        pytest.param(
            [
                ("l1", 1, 1),
                ("l1", 1, 2),
                ("l1", 2, 3),
                ("l1", 2, 4),
                ("l2", 1, 5),
                ("l2", 1, 6),
                ("l2", 2, 7),
                ("l2", 2, 8),
            ],
            6,
            [
                ("l1", 1, 1),
                ("l1", 1, 2),
                ("l1", 2, 3),
                ("l2", 1, 5),
                ("l2", 1, 6),
                ("l2", 2, 7),
            ],
            id="4. Test Subsample Key",
        ),
    ],
)
def test_subsampling(test_metadata, max_files, expected_subsampled_metadata):
    test_metadata = pd.DataFrame(
        test_metadata, columns=["stratify_key", "split_key", "subsample_key"]
    )
    # Shuffle the test_metadata
    test_metadata = test_metadata.sample(frac=1)
    expected_subsampled_metadata = pd.DataFrame(
        expected_subsampled_metadata,
        columns=["stratify_key", "split_key", "subsample_key"],
    )
    subsampled_metadata = subsample_metadata(test_metadata, max_files).reset_index(
        drop=True
    )
    # Test for correctness of subsampling
    assert_frame_equal(subsampled_metadata, expected_subsampled_metadata)
    # Test for stability of subsampling
    assert_frame_equal(
        subsampled_metadata,
        subsample_metadata(test_metadata.sample(frac=1), max_files).reset_index(
            drop=True
        ),
    )
