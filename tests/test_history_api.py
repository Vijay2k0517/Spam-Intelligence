"""Automated tests for prediction history and database pagination endpoints."""

from fastapi import status


def test_history_lifecycle_create_get_filter_delete(client):
    """Verify complete history lifecycle: creation via classify, listing, filtering, and deletion."""
    # 1. Create a spam prediction record
    spam_res = client.post(
        "/api/v1/classify",
        json={"message": "WINNER! Free tickets now. Call 08001234567", "model_id": "hybrid_mnb", "save_history": True},
    )
    assert spam_res.status_code == status.HTTP_200_OK
    spam_id = spam_res.json()["history_id"]
    assert spam_id is not None

    # 2. Create a ham prediction record
    ham_res = client.post(
        "/api/v1/classify",
        json={"message": "Hey friend, are we having lunch today?", "model_id": "text_nb", "save_history": True},
    )
    assert ham_res.status_code == status.HTTP_200_OK
    ham_id = ham_res.json()["history_id"]
    assert ham_id is not None

    # 3. List history records
    list_res = client.get("/api/v1/history?page=1&page_size=10")
    assert list_res.status_code == status.HTTP_200_OK
    data = list_res.json()
    assert data["pagination"]["total_items"] >= 2
    assert len(data["items"]) >= 2

    # 4. Filter by model_id
    model_res = client.get("/api/v1/history?model_id=hybrid_mnb")
    assert model_res.status_code == status.HTTP_200_OK
    for item in model_res.json()["items"]:
        assert item["model_id"] == "hybrid_mnb"

    # 5. Filter by prediction class
    spam_filter_res = client.get("/api/v1/history?prediction=spam")
    assert spam_filter_res.status_code == status.HTTP_200_OK
    for item in spam_filter_res.json()["items"]:
        assert item["prediction"] == "spam"
        assert item["is_spam"] is True

    # 6. Get single record by ID
    single_res = client.get(f"/api/v1/history/{spam_id}")
    assert single_res.status_code == status.HTTP_200_OK
    single_data = single_res.json()
    assert single_data["id"] == spam_id
    assert single_data["model_id"] == "hybrid_mnb"

    # 7. Delete record by ID
    del_res = client.delete(f"/api/v1/history/{spam_id}")
    assert del_res.status_code == status.HTTP_200_OK
    assert del_res.json()["deleted"] is True

    # 8. Verify record is now 404
    after_del_res = client.get(f"/api/v1/history/{spam_id}")
    assert after_del_res.status_code == status.HTTP_404_NOT_FOUND


def test_history_invalid_prediction_filter_rejected(client):
    """Verify invalid prediction filter returns 400 Bad Request."""
    response = client.get("/api/v1/history?prediction=invalid_class")
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_history_nonexistent_record_returns_404(client):
    """Verify non-existent history record returns 404 on get and delete."""
    assert client.get("/api/v1/history/999999").status_code == status.HTTP_404_NOT_FOUND
    assert client.delete("/api/v1/history/999999").status_code == status.HTTP_404_NOT_FOUND
