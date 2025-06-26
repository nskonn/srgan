from unittest.mock import patch

from src.utils.get_last_checkpoint_name import get_last_checkpoint_name


@patch("os.listdir")
def test_get_last_checkpoint_name(mock_listdir):
    mock_listdir.return_value = [
        "checkpoint_001.pth",
        "checkpoint_150.pth",
        "checkpoint_045.pth",
        "notes.txt",
        "checkpoint_099.pth",
    ]

    result = get_last_checkpoint_name()
    assert result == "checkpoint_150.pth"


@patch("os.listdir")
def test_get_last_checkpoint_name_no_checkpoints(mock_listdir):
    mock_listdir.return_value = ["notes.txt", "image.png"]

    result = get_last_checkpoint_name()
    assert result is None
