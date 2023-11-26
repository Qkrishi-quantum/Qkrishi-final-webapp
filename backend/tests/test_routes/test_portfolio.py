from tests.utils.portfolio import create_random_portfolio


def test_should_fetch_portfolio_created(client, db_session):
    portfolio = create_random_portfolio(db=db_session)
    # print(blog.__dict__)
    response = client.get(f"/api/portfolio/{portfolio.id}")
    assert response.status_code == 200
    assert response.json()["holdings"] == portfolio.holdings
