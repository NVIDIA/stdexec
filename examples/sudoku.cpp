/*
    Copyright (c) 2005-2021 Intel Corporation
    Copyright (c) Facebook, Inc. and its affiliates.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Copyright 2026 NVIDIA Corp.
*/

// This sudoku code was originally taken from TBB examples/thread_group/sudoku.
// The TBB example leaks: https://github.com/oneapi-src/oneTBB/issues/568
// The code was modified by Kirk Schoop to use libunifex instead of TBB and to
// remove the leaks.  The code was later modified by David Olsen to use C++26
// std::execution instead of libunifex, to be a test in the NVHPC test suite
// for stdexec.

#include <cstdio>
#include <cstdlib>

#include <atomic>
#include <chrono>
#include <execution>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#define STDEXEC_NAMESPACE std::execution
#include <stdexec/execution.hpp>

unsigned const        BOARD_SIZE = 81;
unsigned const        BOARD_DIM  = 9;
std::atomic<unsigned> nSols;
std::atomic<unsigned> nTasks;
std::atomic<unsigned> nRaces;
bool                  find_one                 = false;
bool                  verbose                  = false;
unsigned short        init_values0[BOARD_SIZE] = {1, 0, 0, 9, 0, 0, 0, 8, 0, 0, 8, 0, 2, 0, 0, 0, 0,
                                                  0, 0, 0, 5, 0, 0, 0, 7, 0, 0, 0, 5, 2, 1, 0, 0, 4,
                                                  0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 7, 4, 0, 0, 7, 0, 0,
                                                  0, 3, 0, 0, 3, 0, 0, 0, 2, 0, 0, 5, 0, 0, 0, 0, 0,
                                                  0, 1, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 0};
unsigned short        init_values1[BOARD_SIZE] = {2, 0, 1, 0, 0, 0, 0, 8, 0, 0, 8, 0, 2, 1, 9, 6, 0,
                                                  0, 0, 0, 5, 0, 0, 0, 7, 0, 0, 0, 5, 2, 1, 0, 0, 4,
                                                  0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 7, 4, 0, 0, 7, 0, 0,
                                                  0, 3, 0, 0, 3, 0, 0, 0, 2, 0, 0, 5, 0, 0, 0, 0, 3,
                                                  0, 1, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 6};
unsigned short        init_values2[BOARD_SIZE] = {1, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                                                  0, 0, 0, 5, 0, 0, 0, 7, 0, 0, 0, 5, 2, 6, 0, 0, 4,
                                                  0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 7, 4, 0, 0, 7, 0, 0,
                                                  0, 3, 0, 0, 3, 0, 0, 0, 2, 0, 0, 5, 0, 0, 0, 0, 0,
                                                  0, 1, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 0};
unsigned short        init_values3[BOARD_SIZE] = {1, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                                                  0, 0, 0, 5, 0, 0, 0, 7, 0, 0, 0, 0, 2, 6, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 7, 4, 0, 0, 0, 0, 0,
                                                  0, 3, 0, 0, 3, 0, 0, 0, 2, 0, 0, 5, 0, 0, 0, 0, 0,
                                                  0, 1, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 0};
unsigned short       *all_init_values[] = {init_values0, init_values1, init_values2, init_values3};

struct board_element
{
  unsigned short solved_element;
  unsigned       potential_set;
};

void print_board(board_element *b)
{
  for (unsigned row = 0; row < BOARD_DIM; ++row)
  {
    for (unsigned col = 0; col < BOARD_DIM; ++col)
    {
      printf(" %d", b[row * BOARD_DIM + col].solved_element);
      if (col == 2 || col == 5)
        printf(" |");
    }
    printf("\n");
    if (row == 2 || row == 5)
      printf(" ---------------------\n");
  }
}

void print_potential_board(board_element *b)
{
  for (unsigned row = 0; row < BOARD_DIM; ++row)
  {
    for (unsigned col = 0; col < BOARD_DIM; ++col)
    {
      if (b[row * BOARD_DIM + col].solved_element)
        printf("  %4d ", b[row * BOARD_DIM + col].solved_element);
      else
        printf(" [%4d]", b[row * BOARD_DIM + col].potential_set);
      if (col == 2 || col == 5)
        printf(" |");
    }
    printf("\n");
    if (row == 2 || row == 5)
      printf(" ----------------------------------------------------------------"
             "--\n");
  }
}

void init_board(board_element *b)
{
  for (unsigned i = 0; i < BOARD_SIZE; ++i)
    b[i].solved_element = b[i].potential_set = 0;
}

void init_board(board_element *b, unsigned short const arr[81])
{
  for (unsigned i = 0; i < BOARD_SIZE; ++i)
  {
    b[i].solved_element = arr[i];
    b[i].potential_set  = 0;
  }
}

void init_potentials(board_element *b)
{
  for (unsigned i = 0; i < BOARD_SIZE; ++i)
    b[i].potential_set = 0;
}

void copy_board(board_element *src, board_element *dst)
{
  for (unsigned i = 0; i < BOARD_SIZE; ++i)
    dst[i].solved_element = src[i].solved_element;
}

bool fixed_board(board_element *b)
{
  for (int i = BOARD_SIZE - 1; i >= 0; --i)
    if (b[i].solved_element == 0)
      return false;
  return true;
}

bool in_row(board_element *b, unsigned row, unsigned col, unsigned short p)
{
  for (unsigned c = 0; c < BOARD_DIM; ++c)
    if (c != col && b[row * BOARD_DIM + c].solved_element == p)
      return true;
  return false;
}

bool in_col(board_element *b, unsigned row, unsigned col, unsigned short p)
{
  for (unsigned r = 0; r < BOARD_DIM; ++r)
    if (r != row && b[r * BOARD_DIM + col].solved_element == p)
      return true;
  return false;
}

bool in_block(board_element *b, unsigned row, unsigned col, unsigned short p)
{
  unsigned b_row = row / 3 * 3, b_col = col / 3 * 3;
  for (unsigned i = b_row; i < b_row + 3; ++i)
    for (unsigned j = b_col; j < b_col + 3; ++j)
      if (!(i == row && j == col) && b[i * BOARD_DIM + j].solved_element == p)
        return true;
  return false;
}

void calculate_potentials(board_element *b)
{
  for (unsigned i = 0; i < BOARD_SIZE; ++i)
  {
    b[i].potential_set = 0;
    if (!b[i].solved_element)
    {  // element is not yet fixed
      unsigned row = i / BOARD_DIM, col = i % BOARD_DIM;
      for (unsigned potential = 1; potential <= BOARD_DIM; ++potential)
      {
        if (!in_row(b, row, col, potential) && !in_col(b, row, col, potential)
            && !in_block(b, row, col, potential))
          b[i].potential_set |= 1 << (potential - 1);
      }
    }
  }
}

bool valid_board(board_element *b)
{
  bool success = true;
  for (unsigned i = 0; i < BOARD_SIZE; ++i)
  {
    if (success && b[i].solved_element)
    {  // element is fixed
      unsigned row = i / BOARD_DIM, col = i % BOARD_DIM;
      if (in_row(b, row, col, b[i].solved_element) || in_col(b, row, col, b[i].solved_element)
          || in_block(b, row, col, b[i].solved_element))
        success = false;
    }
  }
  return success;
}

bool examine_potentials(board_element *b, bool *progress)
{
  bool singletons = false;
  for (unsigned i = 0; i < BOARD_SIZE; ++i)
  {
    if (b[i].solved_element == 0 && b[i].potential_set == 0)  // empty set
      return false;
    switch (b[i].potential_set)
    {
    case 1:
    {
      b[i].solved_element = 1;
      singletons          = true;
      break;
    }
    case 2:
    {
      b[i].solved_element = 2;
      singletons          = true;
      break;
    }
    case 4:
    {
      b[i].solved_element = 3;
      singletons          = true;
      break;
    }
    case 8:
    {
      b[i].solved_element = 4;
      singletons          = true;
      break;
    }
    case 16:
    {
      b[i].solved_element = 5;
      singletons          = true;
      break;
    }
    case 32:
    {
      b[i].solved_element = 6;
      singletons          = true;
      break;
    }
    case 64:
    {
      b[i].solved_element = 7;
      singletons          = true;
      break;
    }
    case 128:
    {
      b[i].solved_element = 8;
      singletons          = true;
      break;
    }
    case 256:
    {
      b[i].solved_element = 9;
      singletons          = true;
      break;
    }
    }
  }
  *progress = singletons;
  return valid_board(b);
}

void spawn_partial_solve(std::execution::parallel_scheduler     sch,
                         std::execution::simple_counting_scope &scope,
                         std::inplace_stop_source              &stop,
                         std::unique_ptr<board_element[]>       board,
                         unsigned                               first_potential_set);

void partial_solve(std::execution::parallel_scheduler     sch,
                   std::execution::simple_counting_scope &scope,
                   std::inplace_stop_source              &stop,
                   std::unique_ptr<board_element[]>       board,
                   unsigned                               first_potential_set)
{
  if (stop.stop_requested())
  {
    ++nRaces;
    return;
  }
  if (fixed_board(board.get()))
  {
    if (++nSols == 1 && verbose)
    {
      print_board(board.get());
    }
    if (find_one)
    {
      stop.request_stop();
    }
    return;
  }
  calculate_potentials(board.get());
  bool progress = true;
  bool success  = examine_potentials(board.get(), &progress);
  if (success && progress)
  {
    partial_solve(sch, scope, stop, std::move(board), first_potential_set);
    return;
  }
  else if (success && !progress)
  {
    while (board.get()[first_potential_set].solved_element != 0)
    {
      ++first_potential_set;
    }
    auto potential_board =
      [=, &scope, &stop, board = std::move(board)](unsigned short potential) mutable noexcept
    {
      if (1 << (potential - 1) & board.get()[first_potential_set].potential_set)
      {
        std::unique_ptr<board_element[]> new_board(new board_element[BOARD_SIZE]);
        copy_board(board.get(), new_board.get());
        new_board.get()[first_potential_set].solved_element = potential;
        spawn_partial_solve(sch, scope, stop, std::move(new_board), first_potential_set);
      }
    };
    for (int i = 1; i < 10; ++i)
    {
      potential_board(i);
    }
  }
}

void spawn_partial_solve(std::execution::parallel_scheduler     sch,
                         std::execution::simple_counting_scope &scope,
                         std::inplace_stop_source              &stop,
                         std::unique_ptr<board_element[]>       board,
                         unsigned                               first_potential_set)
{
  std::execution::spawn(
    std::execution::schedule(sch)
      | std::execution::then(
        [=, &scope, &stop, board = std::move(board)]() mutable noexcept
        {
          ++nTasks;
          partial_solve(sch, scope, stop, std::move(board), first_potential_set);
        })
      | std::execution::upon_error([](auto) noexcept {}),  // no-op, just swallow errors
    scope.get_token());
}

std::tuple<unsigned, unsigned, unsigned, std::chrono::steady_clock::duration>
solve(std::execution::parallel_scheduler sch, unsigned short const *init_values)
{
  nSols  = 0;
  nTasks = 0;
  nRaces = 0;
  std::unique_ptr<board_element[]> start_board{new board_element[BOARD_SIZE]};
  init_board(start_board.get(), init_values);
  auto                     start = std::chrono::steady_clock::now();
  std::inplace_stop_source stop;
  auto                     canceled = []()
  {
    if (verbose)
    {
      printf("\ncanceled \n\n");
    }
  };
  std::inplace_stop_token::template callback_type<decltype(canceled)> callback(stop.get_token(),
                                                                               canceled);
  std::execution::simple_counting_scope                               scope;
  spawn_partial_solve(sch, scope, stop, std::move(start_board), 0);
  std::this_thread::sync_wait(scope.join());
  return std::make_tuple((unsigned) nSols,
                         (unsigned) nTasks,
                         (unsigned) nRaces,
                         std::chrono::steady_clock::now() - start);
}

using double_sec = std::chrono::duration<double>;

int main(int argc, char *argv[])
{
  bool silent = false;

  std::vector<std::string_view> args(argv + 1, argv + argc);
  for (auto arg: args)
  {
    if (arg == "find-one")
    {
      find_one = true;
    }
    else if (arg == "verbose")
    {
      verbose = true;
    }
    else if (arg == "silent")
    {
      silent = true;
    }
    else
    {
      printf("unrecognized argument: -> %s", arg.data());
    }
  }

  if (silent)
    verbose = false;

  auto one_solve = [&](unsigned short const *init_values)
  {
    auto [number, tasks, races, solve_time] = solve(std::execution::get_parallel_scheduler(),
                                                    init_values);
    if (!silent)
    {
      if (find_one)
      {
        printf("Sudoku: Time for first solution using %5u tasks with %5u "
               "races: %9.6fs\n",
               tasks,
               races,
               std::chrono::duration_cast<double_sec>(solve_time).count());
      }
      else
      {
        printf("Sudoku: Time for all %6u solutions using %7u tasks: %9.6fs\n",
               number,
               tasks,
               std::chrono::duration_cast<double_sec>(solve_time).count());
      }
    }
    fflush(stdout);
  };

  for (int iv = 0; iv < 4; ++iv)
  {
    unsigned short const *init_values = all_init_values[iv];
    find_one                          = false;
    one_solve(init_values);
    find_one = true;
    one_solve(init_values);
  }
}
