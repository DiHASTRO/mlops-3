'''
Test for GET /ping endpoint
'''
import pytest


@pytest.mark.parametrize(
    ('params', 'expected_status_code', 'expected_response'),
    [
        pytest.param(
            [],
            200,
            {'status': 'ok'},
        ),
    ],
)
def test_ping(
    client,
    params,
    expected_status_code,
    expected_response,
):
    '''
    Test for GET /ping endpoint
    '''
    response = client.get('/ping/')
    assert response.status_code == expected_status_code
    assert response.json() == expected_response
