from fastapi.testclient import TestClient
import os
import sys

# Add parent directory (2 levels above) to sys.path so main.py can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(
    __file__), '../../')))

from main import app  # noqa: E402

client = TestClient(app)


def test_favicon_exists():
    response = client.get("/favicon.ico")
    assert response.status_code == 200  # Ensure the favicon endpoint exists

    # Check the content type
    assert response.headers["content-type"] in ["image/x-icon",
                                                "image/vnd.microsoft.icon"]
    assert len(response.content) > 0  # Ensure it's not an empty file


def test_favicon_file_exists():
    assert os.path.exists("static/favicon.ico")  # Verify the file exists


def test_favicon_http_methods():
    response_post = client.post("/favicon.ico")
    assert response_post.status_code == 405  # Method not allowed for POST

    response_delete = client.delete("/favicon.ico")
    assert response_delete.status_code == 405  # Method not allowed for DELETE
