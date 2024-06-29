# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#%%
"""Implements a PyGrain DataLoader for chess data."""

import abc
from distutils.command import build
import os

import grain.python as pygrain
import jax
import numpy as np

import bagz
import config as config_lib
import constants
import tokenizer
import math

import chess
import numpy as np


# The lists of the strings of the row and columns of a chess board,
# traditionally named rank and file.
_CHESS_FILE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


def _compute_all_possible_actions() -> tuple[dict[str, int], dict[int, str]]:
  """Returns two dicts converting moves to actions and actions to moves.

  These dicts contain all possible chess moves.
  """
  all_moves = []

  # First, deal with the normal moves.
  # Note that this includes castling, as it is just a rook or king move from one
  # square to another.
  board = chess.BaseBoard.empty()
  for square in range(64):
    next_squares = []

    # Place the queen and see where it attacks (we don't need to cover the case
    # for a bishop, rook, or pawn because the queen's moves includes all their
    # squares).
    board.set_piece_at(square, chess.Piece.from_symbol('Q'))
    next_squares += board.attacks(square)

    # Place knight and see where it attacks
    board.set_piece_at(square, chess.Piece.from_symbol('N'))
    next_squares += board.attacks(square)
    board.remove_piece_at(square)

    for next_square in next_squares:
      all_moves.append(
          chess.square_name(square) + chess.square_name(next_square)
      )

  # Then deal with promotions.
  # Only look at the last ranks.
  promotion_moves = []
  for rank, next_rank in [('2', '1'), ('7', '8')]:
    for index_file, file in enumerate(_CHESS_FILE):
      # Normal promotions.
      move = f'{file}{rank}{file}{next_rank}'
      promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]

      # Capture promotions.
      # Left side.
      if file > 'a':
        next_file = _CHESS_FILE[index_file - 1]
        move = f'{file}{rank}{next_file}{next_rank}'
        promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
      # Right side.
      if file < 'h':
        next_file = _CHESS_FILE[index_file + 1]
        move = f'{file}{rank}{next_file}{next_rank}'
        promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
  all_moves += promotion_moves

  move_to_action, action_to_move = {}, {}
  for action, move in enumerate(all_moves):
    assert move not in move_to_action
    move_to_action[move] = action
    action_to_move[action] = move

  return move_to_action, action_to_move


MOVE_TO_ACTION, ACTION_TO_MOVE = _compute_all_possible_actions()
NUM_ACTIONS = len(MOVE_TO_ACTION)


def centipawns_to_win_probability(centipawns: int) -> float:
  """Returns the win probability (in [0, 1]) converted from the centipawn score.

  Reference: https://lichess.org/page/accuracy
  Well-known transformation, backed by real-world data.

  Args:
    centipawns: The chess score in centipawns.
  """
  return 0.5 + 0.5 * (2 / (1 + math.exp(-0.00368208 * centipawns)) - 1)


def get_uniform_buckets_edges_values(
    num_buckets: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns edges and values of uniformly sampled buckets in [0, 1].

  Example: for num_buckets=4, it returns:
  edges=[0.25, 0.50, 0.75]
  values=[0.125, 0.375, 0.625, 0.875]

  Args:
    num_buckets: Number of buckets to create.
  """
  full_linspace = np.linspace(0.0, 1.0, num_buckets + 1)
  edges = full_linspace[1:-1]
  values = (full_linspace[:-1] + full_linspace[1:]) / 2
  return edges, values


def compute_return_buckets_from_returns(
    returns: np.ndarray,
    bins_edges: np.ndarray,
) -> np.ndarray:
  """Arranges the discounted returns into bins.

  The returns are put into the bins specified by `bin_edges`. The length of
  `bin_edges` is equal to the number of buckets minus 1. In case of a tie (if
  the return is exactly equal to an edge), we take the bucket right before the
  edge. See example below.
  This function is purely using np.searchsorted, so it's a good reference to
  look at.

  Examples:
  * bin_edges=[0.5] and returns=[0., 1.] gives the buckets [0, 1].
  * bin_edges=[-30., 30.] and returns=[-200., -30., 0., 1.] gives the buckets
    [0, 0, 1, 1].

  Args:
    returns: An array of discounted returns, rank 1.
    bins_edges: The boundary values of the return buckets, rank 1.

  Returns:
    An array of buckets, described as integers, rank 1.

  Raises:
    ValueError if `returns` or `bins_edges` are not of rank 1.
  """
  if len(returns.shape) != 1:
    raise ValueError(
        'The passed returns should be of rank 1. Got'
        f' rank={len(returns.shape)}.'
    )
  if len(bins_edges.shape) != 1:
    raise ValueError(
        'The passed bins_edges should be of rank 1. Got'
        f' rank{len(bins_edges.shape)}.'
    )
  return np.searchsorted(bins_edges, returns, side='left')


def _process_fen(fen: str) -> np.ndarray:
  return tokenizer.tokenize(fen).astype(np.int32)  # type: ignore


def _process_move(move: str) -> np.ndarray:
  return np.asarray([MOVE_TO_ACTION[move]], dtype=np.int32)


def _process_win_prob(
    win_prob: float,
    return_buckets_edges: np.ndarray,
) -> np.ndarray:
  return compute_return_buckets_from_returns(
      returns=np.asarray([win_prob]),
      bins_edges=return_buckets_edges,
  )


class ConvertToSequence(pygrain.MapTransform, abc.ABC):
  """Base class for converting chess data to a sequence of integers."""

  def __init__(self, num_return_buckets: int) -> None:
    super().__init__()
    self._return_buckets_edges, _ = get_uniform_buckets_edges_values(
        num_return_buckets,
    )
    # The loss mask ensures that we only train on the return bucket.
    self._loss_mask = np.full(
        shape=(self._sequence_length,),
        fill_value=True,
        dtype=bool,
    )
    self._loss_mask[-1] = False

  @property
  @abc.abstractmethod
  def _sequence_length(self) -> int:
    raise NotImplementedError()


class ConvertStateValueDataToSequence(ConvertToSequence):
  """Converts the fen, move, and win probability into a sequence of integers."""

  @property
  def _sequence_length(self) -> int:
    return tokenizer.SEQUENCE_LENGTH + 1  # (s) +  (r)

  def map( self, element: bytes):
    fen, win_prob = constants.CODERS['state_value'].decode(element)
    state = _process_fen(fen)
    return_bucket = _process_win_prob(win_prob, self._return_buckets_edges)
    sequence = np.concatenate([state, return_bucket])
    return sequence, self._loss_mask  # type: ignore


_TRANSFORMATION_BY_POLICY = {
    'state_value': ConvertStateValueDataToSequence,
}


# Follows the base_constants.DataLoaderBuilder protocol.
def build_data_loader(path: str, config: config_lib.DataConfig) -> pygrain.DataLoader:
  """Returns a data loader for chess from the config."""
  data_source = bagz.BagDataSource(path)

  if config.num_records is not None:
    num_records = config.num_records
    if len(data_source) < num_records:
      raise ValueError(
          f'[Process {jax.process_index()}]: The number of records requested'
          f' ({num_records}) is larger than the dataset ({len(data_source)}).'
      )
  else:
    num_records = len(data_source)

  sampler = pygrain.IndexSampler(
      num_records=num_records,
      shard_options=pygrain.NoSharding(),
      shuffle=config.shuffle,
      num_epochs=None,
      seed=config.seed,
  )
  transformations = (
      _TRANSFORMATION_BY_POLICY[config.policy](
          num_return_buckets=config.num_return_buckets
      ),
      pygrain.Batch(config.batch_size, drop_remainder=True),
  )
  return pygrain.DataLoader(
      data_source=data_source,
      sampler=sampler,
      operations=transformations,
      worker_count=config.worker_count,
      read_options=None,
  )
config = config_lib.DataConfig(
    batch_size=4,
    shuffle=True,
    seed=0,
    drop_remainder=True,
    worker_count=0,
    num_return_buckets=16,
    split='test',
    policy='state_value',
    num_records=None,
)
dataloader = build_data_loader("/root/chess-hackathon-fork/test_data/state_value_data.bag", config)
first_batch = next(iter(dataloader))