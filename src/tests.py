from utils import (
    generate_weighted_similarity_matrix,
    get_recommendations,
    load_data,
    load_similarity_matrices,
)
import config


def test_similarity_matrices():
    (
        cast_similarity,
        director_similarity,
        keywords_similarity,
        overview_similarity,
        user_embedding_similarity,
    ) = load_similarity_matrices()
    data = load_data(config.DATA_PATH)

    sim_matrices = [
        cast_similarity,
        director_similarity,
        keywords_similarity,
        overview_similarity,
        user_embedding_similarity,
    ]

    sim_weights = [1, 1, 1, 1, 5]
    similarity_matrix = generate_weighted_similarity_matrix(
        arrays=sim_matrices, weights=sim_weights
    )

    recommendations = get_recommendations(
        films=data,
        titles=["Spirited Away", "Howl's Moving Castle"],
        similarity_matrix=similarity_matrix,
        top_n=4,
    )
    assert recommendations == [
        "Princess Mononoke",
        "Nausicaä of the Valley of the Wind",
        "The Wind Rises",
        "Castle in the Sky",
    ]
