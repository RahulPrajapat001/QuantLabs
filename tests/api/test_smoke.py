def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ok"


def test_securities(client):
    r = client.get("/securities")
    assert r.status_code == 200, r.text
    assert r.json()["total"] == 8


def test_logged_run_fixture(client, logged_run):
    r = client.get(f"/research/log/{logged_run}")
    assert r.status_code == 200, r.text
    assert r.json()["sharpe"]["trials"] >= 1
