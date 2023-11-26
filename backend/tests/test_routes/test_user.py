def test_create_user(client):
    data = {"email": "testuser@nofoobar.com", "password": "testing", "name": "Mr. Unnamed"}
    response = client.post("/api/users", json=data)
    assert response.status_code == 201
    assert response.json()["email"] == "testuser@nofoobar.com"
    assert response.json()["name"] == "Mr. Unnamed"
