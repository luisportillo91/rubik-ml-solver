"""Rubik package: Pure Python functional representation, search, and ML encoder."""

from rubik.cube import (
    GETTERS,
    GETTER_LIST,
    INVERSE_MOVES,
    MOVE_TO_INDEX,
    PERMUTATIONS,
    SOLVED_FACELET,
    VALID_MOVES,
    apply_move,
    apply_move_batch,
    apply_moves,
    compose_moves,
    create_solved_cube,
    from_facelet_string,
    scramble,
    to_facelet_string,
)
from rubik.dataset import (
    ContrastiveCubeIterableDataset,
    generate_contrastive_batch,
    generate_contrastive_sample,
)
from rubik.encoder import (
    COLOR_TO_ID,
    CubeTransformerEncoder,
    ID_TO_COLOR,
    batch_to_tensor,
    cube_to_tensor,
)
from rubik.loss import InfoNCELoss
from rubik.search import (
    create_distance_filter,
    find_distance,
    find_path,
    generate_neighborhood_tree,
    get_neighbors,
    is_within_distance,
)

__all__ = [
    "VALID_MOVES",
    "INVERSE_MOVES",
    "SOLVED_FACELET",
    "PERMUTATIONS",
    "GETTERS",
    "GETTER_LIST",
    "MOVE_TO_INDEX",
    "create_solved_cube",
    "to_facelet_string",
    "from_facelet_string",
    "apply_move",
    "apply_moves",
    "compose_moves",
    "apply_move_batch",
    "scramble",
    "get_neighbors",
    "generate_neighborhood_tree",
    "is_within_distance",
    "create_distance_filter",
    "find_distance",
    "find_path",
    "CubeTransformerEncoder",
    "cube_to_tensor",
    "batch_to_tensor",
    "COLOR_TO_ID",
    "ID_TO_COLOR",
    "InfoNCELoss",
    "generate_contrastive_sample",
    "generate_contrastive_batch",
    "ContrastiveCubeIterableDataset",
]
