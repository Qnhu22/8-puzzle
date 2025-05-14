import pygame
import sys
import random
from collections import deque
import heapq
import timeit
import math
from collections import defaultdict
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from collections import defaultdict
import time
import pickle

def draw_performance_plotly(performance_history):
        algorithms = []
        avg_states_explored = []
        avg_runtimes = []

        for algo, runs in performance_history.items():
            if runs:
                algorithms.append(algo)
                states = [run["states_explored"] if isinstance(run["states_explored"], (int, float)) else len(run["states_explored"]) for run in runs]
                avg_states_explored.append(sum(states) / len(states))
                runtimes = [run["runtime"] for run in runs]
                avg_runtimes.append(sum(runtimes) / len(runtimes))

        if not algorithms:
            print("No performance data available.")
            return

        df = pd.DataFrame({
            'Algorithm': algorithms,
            'States Explored': avg_states_explored,
            'Runtime (ms)': avg_runtimes
        })

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                            subplot_titles=("States Explored", "Runtime (ms)"))

        fig.add_trace(go.Bar(
            x=df["Algorithm"],
            y=df["States Explored"],
            text=df["States Explored"],  # Hiển thị giá trị
            textposition='auto',
            name="States Explored",
            marker_color='darkblue'
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=df["Algorithm"],
            y=df["Runtime (ms)"],
            text=[f"{val:.2f}" for val in df["Runtime (ms)"]],  # Làm tròn 2 chữ số
            textposition='auto',
            name="Runtime (ms)",
            marker_color='hotpink'
        ), row=2, col=1)


        fig.update_layout(height=600, width=900, title_text="Performance Comparison", showlegend=False)
        fig.update_yaxes(title_text="States", row=1, col=1)
        fig.update_yaxes(title_text="Time (ms)", row=2, col=1)

        fig.show()  # 👉 Mở trình duyệt và hiển thị biểu đồ HTML
class EightPuzzle:
    def __init__(self, initial, goal):
        self.initial = tuple(map(tuple, initial))
        self.goal = tuple(map(tuple, goal))
        self.rows, self.cols = 3, 3

    def find_blank(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j
        return None

    def get_neighbors(self, state):
        row, col = self.find_blank(state)
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        state_list = [list(row) for row in state]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = [row[:] for row in state_list]
                new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                moves.append(tuple(map(tuple, new_state)))
        return moves

    def state_to_list(self, state):
        return [num for row in state for num in row]

    def list_to_state(self, lst):
        return tuple(tuple(lst[i * 3:(i + 1) * 3]) for i in range(3))

    def generate_random_state(self):
        numbers = list(range(9))
        random.shuffle(numbers)
        state = self.list_to_state(numbers)
        while not self.is_solvable(state):
            random.shuffle(numbers)
            state = self.list_to_state(numbers)
        return state

    def fitness(self, state):
        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] and state[i][j] != self.goal[i][j]:
                    value = state[i][j]
                    goal_i, goal_j = divmod(value - 1, 3)
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return -distance

    def crossover(self, parent1, parent2):
        p1_list = self.state_to_list(parent1)
        p2_list = self.state_to_list(parent2)
        crossover_point = random.randint(1, 7)
        child = p1_list[:crossover_point]
        seen = set(child)
        for num in p2_list:
            if num not in seen:
                child.append(num)
                seen.add(num)
        return self.list_to_state(child)

    def mutate(self, state, mutation_rate=0.05):
        state_list = self.state_to_list(state)
        if random.random() < mutation_rate:
            i, j = random.sample(range(9), 2)
            state_list[i], state_list[j] = state_list[j], state_list[i]
        return self.list_to_state(state_list)

    def reconstruct_path(self, final_state):
        path = [self.initial]
        current = self.initial
        while current != final_state:
            neighbors = self.get_neighbors(current)
            current = min(neighbors, key=lambda x: self.heuristic(x), default=final_state)
            path.append(current)
            if current == final_state:
                break
        return path

    def genetic_algorithm(self, population_size=50, max_generations=500):
        population = [self.generate_random_state() for _ in range(population_size)]
        explored_states = []
        best_fitness = float('-inf')
        no_improvement_count = 0
        max_no_improvement = 100

        for generation in range(max_generations):
            population = sorted(population, key=self.fitness, reverse=True)
            explored_states.extend(population[:5])
            best_state = population[0]
            current_fitness = self.fitness(best_state)

            if best_state == self.goal:
                path = self.reconstruct_path(best_state)
                return path, explored_states

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= max_no_improvement:
                break

            new_population = population[:population_size // 2]
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(new_population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population

        return None, explored_states

    def bfs(self):
        queue = deque([(self.initial, [])])
        visited = {self.initial}
        explored_states = []

        while queue:
            state, path = queue.popleft()
            explored_states.append(state)
            if state == self.goal:
                return path + [state], explored_states
            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [state]))
        return None, explored_states

    def dfs(self, depth_limit=1000):
        stack = [(self.initial, [])]
        visited = {self.initial}
        explored_states = []

        while stack:
            state, path = stack.pop()
            explored_states.append(state)
            if state == self.goal:
                return path + [state], explored_states
            if len(path) < depth_limit:
                for neighbor in self.get_neighbors(state):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append((neighbor, path + [state]))
        return None, explored_states

    def ucs(self):
        pq = [(0, self.initial, [])]
        visited = {self.initial: 0}
        explored_states = []

        while pq:
            cost, state, path = heapq.heappop(pq)
            explored_states.append(state)
            if state == self.goal:
                return path + [state], explored_states
            for neighbor in self.get_neighbors(state):
                new_cost = cost + 1
                if neighbor not in visited or new_cost < visited[neighbor]:
                    visited[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor, path + [state]))
        return None, explored_states

    def ids(self, max_depth=100):
        for depth in range(max_depth + 1):
            stack = [(self.initial, [], 0)]
            visited = set()
            explored_states = []
            while stack:
                state, path, current_depth = stack.pop()
                if state not in visited:
                    visited.add(state)
                    explored_states.append(state)
                    if state == self.goal:
                        return path + [state], explored_states
                    if current_depth < depth:
                        for neighbor in self.get_neighbors(state):
                            if neighbor not in visited:
                                stack.append((neighbor, path + [state], current_depth + 1))
            if explored_states and explored_states[-1] == self.goal:
                return path + [state], explored_states
        return None, explored_states

    def greedy(self):
        pq = [(self.heuristic(self.initial), self.initial, [])]
        visited = {self.initial}
        explored_states = []

        while pq:
            _, state, path = heapq.heappop(pq)
            explored_states.append(state)
            if state == self.goal:
                return path + [state], explored_states
            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    heapq.heappush(pq, (self.heuristic(neighbor), neighbor, path + [state]))
        return None, explored_states

    def a_star(self, timeout=10.0):
        start_time = timeit.default_timer()
        pq = [(self.heuristic(self.initial), 0, self.initial, [])]
        visited = {}
        explored_states = []

        while pq:
            if timeit.default_timer() - start_time > timeout:
                print("A* timeout after", timeout, "seconds")
                return None, explored_states

            f, g, state, path = heapq.heappop(pq)
            if state not in visited or g < visited[state]:
                visited[state] = g
                explored_states.append(state)

                if state == self.goal:
                    return path + [state], explored_states

                for neighbor in self.get_neighbors(state):
                    new_g = g + 1
                    new_f = new_g + self.heuristic(neighbor)
                    heapq.heappush(pq, (new_f, new_g, neighbor, path + [state]))
        return None, explored_states

    def ida_star(self, timeout=10.0):
        start_time = timeit.default_timer()
        threshold = self.heuristic(self.initial)
        explored_states = []

        while True:
            if timeit.default_timer() - start_time > timeout:
                print("IDA* timeout after", timeout, "seconds")
                return None, explored_states

            result, new_threshold = self.ida_star_recursive(self.initial, [], 0, threshold, explored_states)
            if result:
                return result, explored_states
            if new_threshold == float('inf'):
                return None, explored_states
            threshold = new_threshold
    
    def ida_star_recursive(self, state, path, g, threshold, explored_states):
        f = g + self.heuristic(state)
        explored_states.append(state)

        if f > threshold:
            return None, f
        if state == self.goal:
            return path + [state], threshold

        min_threshold = float('inf')
        for neighbor in self.get_neighbors(state):
            if neighbor not in path:  # tránh lặp vô hạn
                result, new_threshold = self.ida_star_recursive(neighbor, path + [state], g + 1, threshold, explored_states)
                if result:
                    return result, threshold
                min_threshold = min(min_threshold, new_threshold)

        return None, min_threshold



    def simple_hc(self):  # Simple Hill Climbing
        current = self.initial
        path = [current]
        explored_states = [current]
        while True:
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break
            next_state = min(neighbors, key=self.heuristic)
            if self.heuristic(next_state) >= self.heuristic(current):
                break
            current = next_state
            path.append(current)
            explored_states.append(current)
            if current == self.goal:
                return path, explored_states
        return None, explored_states

    def steepest_hc(self):  # Steepest Ascent Hill Climbing
        current = self.initial
        path = [current]
        explored_states = [current]
        while True:
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break
            next_state = min(neighbors, key=self.heuristic)
            if self.heuristic(next_state) >= self.heuristic(current):
                break
            current = next_state
            path.append(current)
            explored_states.append(current)
            if current == self.goal:
                return path, explored_states
        return None, explored_states

    def random_hc(self, max_steps=1000):  # Random Hill Climbing
        current = self.initial
        path = [current]
        explored_states = [current]
        for _ in range(max_steps):
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break
            next_state = random.choice(neighbors)
            if self.heuristic(next_state) < self.heuristic(current):
                current = next_state
                path.append(current)
                explored_states.append(current)
            if current == self.goal:
                return path, explored_states
        return None, explored_states

    def simulated_annealing(self, initial_temp=1000.0, cooling_rate=0.99, min_temp=0.01, max_no_improvement=2000):
        current = self.initial
        path = [current]
        explored_states = set()
        current_heuristic = self.heuristic(current)
        best_state = current
        best_heuristic = current_heuristic
        no_improvement_count = 0
        temperature = initial_temp

        while temperature > min_temp and no_improvement_count < max_no_improvement:
            explored_states.add(current)
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break

            neighbor_heuristic_pairs = [(neighbor, self.heuristic(neighbor)) for neighbor in neighbors]
            neighbor_heuristic_pairs.sort(key=lambda x: x[1])
            next_state, next_heuristic = neighbor_heuristic_pairs[0]

            if next_heuristic >= current_heuristic:
                delta = next_heuristic - current_heuristic
                acceptance_probability = math.exp(-delta / temperature)
                if random.uniform(0, 1) > acceptance_probability:
                    next_state, next_heuristic = random.choice(neighbor_heuristic_pairs)

            if next_heuristic < current_heuristic:
                no_improvement_count = 0
                if next_heuristic < best_heuristic:
                    best_state = next_state
                    best_heuristic = next_heuristic
            else:
                no_improvement_count += 1

            current = next_state
            current_heuristic = next_heuristic
            path.append(current)

            if current == self.goal:
                return path, list(explored_states)

            temperature *= cooling_rate

        if best_state == self.goal:
            return self.reconstruct_path(best_state), list(explored_states)
        return None, list(explored_states)

    def beam_search(self, beam_width=5):  # Tăng beam_width để mở rộng tìm kiếm
        initial_state = self.initial
        if initial_state == self.goal:
            return [initial_state], []

        current_states = [initial_state]
        path = {initial_state: []}
        explored_states = set()

        while current_states:
            next_states = []
            for state in current_states:
                explored_states.add(state)
                neighbors = self.get_neighbors(state)
                for neighbor in neighbors:
                    if neighbor not in path:
                        path[neighbor] = path[state] + [state]
                        next_states.append(neighbor)

            evaluated = [(self.heuristic(state), state) for state in next_states]
            evaluated.sort(key=lambda x: x[0])

            current_states = [state for (_, state) in evaluated[:beam_width]]

            if self.goal in current_states:
                return path[self.goal] + [self.goal], list(explored_states)

        return None, list(explored_states)


    def and_or_search(self, max_steps=1000):
        queue = deque([(self.initial, [], {self.initial}, None)])  # (state, path, and_group, action)
        visited = set()
        explored_states = []
        num_steps = 0

        while queue and num_steps < max_steps:
            state, path, and_group, action = queue.popleft()
            explored_states.append(state)
            num_steps += 1

            # Kiểm tra nhánh AND: Tất cả trạng thái trong and_group phải là mục tiêu
            if all(s == self.goal for s in and_group):
                state_path = [p for p, a in path] + [state]
                return state_path, explored_states

            state_tuple = frozenset(and_group)
            if state_tuple in visited:
                continue
            visited.add(state_tuple)

            # Nhánh OR: Lựa chọn giữa các hành động (lên, xuống, trái, phải)
            neighbors = self.get_neighbors(state)
            for action_idx, neighbor in enumerate(neighbors):
                # Nhánh AND: Tạo tất cả trạng thái có thể xảy ra sau hành động
                and_states = {neighbor}  # Trạng thái bình thường

                # Tạo trạng thái không xác định với xác suất 50%
                if random.random() < 0.7:  # Xác suất 50%
                    i, j = self.find_blank(neighbor)
                    directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
                    valid_directions = [(di, dj) for di, dj in directions if 0 <= i + di < 3 and 0 <= j + dj < 3]
                    for di, dj in valid_directions:
                        ni, nj = i + di, j + dj
                        state_list = list(map(list, neighbor))
                        state_list[i][j], state_list[ni][nj] = state_list[ni][nj], state_list[i][j]
                        uncertain_state = tuple(map(tuple, state_list))
                        and_states.add(uncertain_state)

                # Thêm nhánh AND vào hàng đợi (không thu hẹp and_states)
                if and_states:
                    action_name = ["up", "down", "right", "left"][action_idx]
                    queue.append((neighbor, path + [(state, action_name)], and_states, action_name))

        return None, explored_states

    def generate_random_state(self, max_depth=5):
        """
        Tạo trạng thái ngẫu nhiên gần initial_state trong max_depth bước.
        """
        current_state = self.initial
        for _ in range(random.randint(1, max_depth)):
            neighbors = self.get_neighbors(current_state)
            current_state = random.choice(neighbors)
        return current_state

    def bfs_for_belief(self, start_state, max_depth=10):
        """
        Chạy BFS từ trạng thái start_state để tìm các trạng thái lân cận trong max_depth bước.
        Trả về: Tập hợp các trạng thái lân cận (giới hạn tối đa 5 trạng thái).
        """
        queue = deque([(start_state, 0)])
        visited = {start_state}
        states = set()

        while queue and len(states) < 5:  # Giới hạn số trạng thái
            state, depth = queue.popleft()
            if depth < max_depth:
                for neighbor in self.get_neighbors(state):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        states.add(neighbor)
        return states

    def belief_state_search(self, initial_belief, max_steps=5000):
        initial_belief = set(initial_belief)
        explored = set()
        num_explored_states = 0
        belief_states_path = [list(initial_belief)]
        total_steps = 0

        # Khởi tạo explored states
        for state in initial_belief:
            explored.add(state)
            num_explored_states += 1

        belief_queue = deque([(initial_belief, [])])
        visited = set()

        while belief_queue and num_explored_states < max_steps:
            belief_state, path = belief_queue.popleft()
            belief_state_tuple = frozenset(belief_state)

            # Kiểm tra mục tiêu: Tất cả trạng thái trong belief state phải là goal
            if all(state == self.goal for state in belief_state):
                total_steps = 0
                for initial_state in initial_belief:
                    self.initial = initial_state
                    solution, _ = self.bfs()
                    if solution:
                        total_steps += len(solution) - 1
                    else:
                        return None, explored, 0
                belief_states_path.append([self.goal] * len(initial_belief))
                return belief_states_path, explored, total_steps

            if belief_state_tuple in visited:
                continue
            visited.add(belief_state_tuple)

            # Duyệt qua các hành động (lên, xuống, trái, phải)
            for action in range(4):
                new_belief = set()
                for state in belief_state:
                    neighbors = self.get_neighbors(state)
                    if action < len(neighbors):
                        # Thêm trạng thái xác định
                        next_state = neighbors[action]
                        new_belief.add(next_state)

                        # Tạo trạng thái không xác định (chỉ với xác suất 10%)
                        if random.random() < 0.1:
                            i, j = None, None
                            for r in range(3):
                                for c in range(3):
                                    if next_state[r][c] == 0:
                                        i, j = r, c
                                        break

                            directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # Lên, xuống, phải, trái
                            valid_directions = [(di, dj) for di, dj in directions if
                                                0 <= i + di < 3 and 0 <= j + dj < 3]
                            if valid_directions:
                                di, dj = random.choice(valid_directions)
                                ni, nj = i + di, j + dj
                                state_list = [list(row) for row in next_state]
                                state_list[i][j], state_list[ni][nj] = state_list[ni][nj], state_list[i][j]
                                uncertain_state = tuple(tuple(row) for row in state_list)
                                new_belief.add(uncertain_state)
                    else:
                        # Nếu hành động không hợp lệ, giữ nguyên trạng thái
                        new_belief.add(state)

                # Thu hẹp belief state: Chỉ giữ 3 trạng thái gần mục tiêu nhất
                if new_belief:
                    new_belief = set(sorted(new_belief, key=self.heuristic)[:3])  # Giữ 3 trạng thái tốt nhất

                    for state in new_belief:
                        if state not in explored:
                            explored.add(state)
                            num_explored_states += 1
                    belief_queue.append((new_belief, path + [min(belief_state, key=self.heuristic)]))
                    belief_states_path.append(list(new_belief))

        return None, explored, 0

    def optimized_bfs_for_belief(self, start_state, max_depth=1):
        queue = deque([(start_state, 0)])
        visited = {start_state}
        states = [(self.heuristic(start_state), start_state)]

        while queue:
            state, depth = queue.popleft()
            if depth < max_depth:
                for neighbor in self.get_neighbors(state):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        states.append((self.heuristic(neighbor), neighbor))

        states.sort()
        return {state for _, state in states[:10]}



    def get_observation(self, state):
        """
        Giả lập quan sát một phần: chỉ quan sát được vị trí của ô số 1.
        Trả về vị trí (row, col) của ô số 1 trong trạng thái.
        """
        for i in range(3):
            for j in range(3):
                if state[i][j] == 1:
                    return (i, j)
        return None

    def find_states_with_one_at_00(self, start_state, max_states=3):  # Giảm từ 6 xuống 3
        """
        Tìm các trạng thái có số 1 ở vị trí (0,0) bằng BFS.
        start_state: Trạng thái ban đầu.
        max_states: Số trạng thái tối đa cần tìm là 3.
        Trả về: Danh sách các trạng thái (dạng tuple) có số 1 ở (0,0).
        """
        queue = deque([(start_state, [])])
        visited = {start_state}
        states_with_one_at_00 = []

        while queue and len(states_with_one_at_00) < max_states:
            state, path = queue.popleft()
            if self.get_observation(state) == (0, 0):
                states_with_one_at_00.append(state)
                if len(states_with_one_at_00) >= max_states:
                    break
            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [state]))

        while len(states_with_one_at_00) < max_states:
            numbers = list(range(9))
            random.shuffle(numbers)
            numbers[0] = 1
            remaining_numbers = [num for num in numbers[1:] if num != 1]
            if len(remaining_numbers) < 8:
                remaining_numbers.append(0)
            numbers = [1] + remaining_numbers[:8]
            state = self.list_to_state(numbers)
            if self.is_solvable(state) and state not in states_with_one_at_00:
                states_with_one_at_00.append(state)

        return states_with_one_at_00[:max_states]

    def partial_observable_search(self):
        """
        Partial Observable Search: Tìm kiếm trên không gian belief states với số 1 ở (0,0).
        Trả về: (belief_states_path, explored_states, total_steps) hoặc (None, explored_states, 0).
        """
        # Khởi tạo initial_belief với 3 trạng thái có số 1 ở (0,0)
        initial_belief = self.find_states_with_one_at_00(self.initial, max_states=3)
        queue = deque([(set(initial_belief), [], 0)])  # (belief_state, path, steps)
        visited = set()
        explored_states = []
        belief_states_path = [list(initial_belief)]
        max_steps = 1000

        while queue and len(queue) < max_steps:
            belief_state, path, steps = queue.popleft()
            belief_state_tuple = frozenset(belief_state)
            explored_states.extend(belief_state)

            # Kiểm tra điều kiện mục tiêu: Tất cả trạng thái trong belief state phải là goal
            if all(state == self.goal for state in belief_state):
                total_steps = steps  # Số bước là số hành động (steps)
                belief_states_path.append([self.goal] * 3)  # Chỉ 3 trạng thái
                return belief_states_path, explored_states, total_steps

            if belief_state_tuple in visited:
                continue
            visited.add(belief_state_tuple)

            # Duyệt qua các hành động (lên, xuống, trái, phải)
            for action in range(4):
                new_belief = set()
                for state in belief_state:
                    neighbors = self.get_neighbors(state)
                    if action < len(neighbors):
                        # Thêm trạng thái xác định
                        next_state = neighbors[action]
                        # Chỉ giữ trạng thái có số 1 ở (0,0)
                        if self.get_observation(next_state) == (0, 0):
                            new_belief.add(next_state)

                        # Tạo trạng thái không xác định (chỉ với xác suất 10%)
                        if random.random() < 0.1:  # Giảm từ 50% xuống 10%
                            i, j = self.find_blank(next_state)
                            directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
                            valid_directions = [(di, dj) for di, dj in directions if
                                                0 <= i + di < 3 and 0 <= j + dj < 3]
                            if valid_directions:
                                di, dj = random.choice(valid_directions)
                                ni, nj = i + di, j + dj
                                state_list = [list(row) for row in next_state]
                                state_list[i][j], state_list[ni][nj] = state_list[ni][nj], state_list[i][j]
                                uncertain_state = tuple(tuple(row) for row in state_list)
                                # Chỉ giữ trạng thái không xác định có số 1 ở (0,0)
                                if self.get_observation(uncertain_state) == (0, 0):
                                    new_belief.add(uncertain_state)

                # Thu hẹp belief state: Giữ tối đa 3 trạng thái tốt nhất theo heuristic
                if new_belief:
                    new_belief = set(sorted(new_belief, key=self.heuristic)[:3])  # Giảm từ 5 xuống 3
                    queue.append((new_belief, path + [min(belief_state, key=self.heuristic)], steps + 1))
                    belief_states_path.append(list(new_belief))

        return None, explored_states, 0

    def is_valid_assignment(self, state, pos, value):
        """
        Kiểm tra xem việc gán giá trị cho ô pos có thỏa mãn các ràng buộc không.
        state: Ma trận hiện tại (có thể chứa None).
        pos: Vị trí ô (i,j).
        value: Giá trị cần gán (0-8).
        Trả về: True nếu hợp lệ, False nếu không.
        """
        i, j = pos
        # Ràng buộc: Ô (0,0) phải là 1
        if i == 0 and j == 0 and value != 1:
            return False

        # Ràng buộc: Mỗi số chỉ xuất hiện một lần
        for r in range(3):
            for c in range(3):
                if (r, c) != pos and state[r][c] == value:
                    return False

        # Ràng buộc theo hàng: ô(i,j+1) = ô(i,j) + 1 (trừ ô trống)
        if j > 0 and state[i][j - 1] is not None and value != 0 and state[i][j - 1] != value - 1:
            return False
        if j < 2 and value != 0 and state[i][j + 1] is not None and state[i][j + 1] != value + 1:
            return False

        # Ràng buộc theo cột: ô(i+1,j) = ô(i,j) + 3 (trừ ô trống)
        if i > 0 and state[i - 1][j] is not None and value != 0 and state[i - 1][j] != value - 3:
            return False
        if i < 2 and value != 0 and state[i + 1][j] is not None and state[i + 1][j] != value + 3:
            return False

        return True

    def is_solvable(self, state):
        """
        Kiểm tra xem ma trận có solvable không (số nghịch đảo chẵn).
        state: Ma trận 3x3 (có thể chứa None).
        """
        flat = [state[i][j] for i in range(3) for j in range(3) if state[i][j] is not None and state[i][j] != 0]
        inversions = 0
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if flat[i] > flat[j]:
                    inversions += 1
        return inversions % 2 == 0

    def q_learning_search(self, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.3, decay=0.995, max_steps=100):
        q_table = {}
        rewards_per_episode = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def find_blank(state):
            for i in range(3):
                for j in range(3):
                    if state[i][j] == 0:
                        return i, j

        def get_neighbors(state):
            i, j = find_blank(state)
            neighbors = []
            for idx, (di, dj) in enumerate(directions):
                ni, nj = i + di, j + dj
                if 0 <= ni < 3 and 0 <= nj < 3:
                    new_state = [list(row) for row in state]
                    new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
                    neighbors.append((tuple(map(tuple, new_state)), idx))
            return neighbors

        def heuristic(state):
            dist = 0
            for i in range(3):
                for j in range(3):
                    val = state[i][j]
                    if val == 0:
                        continue
                    gi, gj = divmod(val - 1, 3)
                    dist += abs(i - gi) + abs(j - gj)
            return dist

        initial_state = self.initial
        goal_state = self.goal
        total_states_explored = 0

        for ep in range(episodes):
            state = initial_state
            total_reward = 0
            if state not in q_table:
                q_table[state] = np.zeros(4)
            for step in range(max_steps):
                if np.random.rand() < epsilon:
                    action = np.random.randint(4)
                else:
                    action = np.argmax(q_table[state])

                neighbors = get_neighbors(state)
                next_state = None
                for n, a in neighbors:
                    if a == action:
                        next_state = n
                        break

                if not next_state:
                    reward = -10
                    next_state = state
                else:
                    reward = -heuristic(next_state)
                    if next_state == goal_state:
                        reward = 100

                if next_state not in q_table:
                    q_table[next_state] = np.zeros(4)

                q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
                total_reward += reward
                state = next_state
                total_states_explored += 1
                if state == goal_state:
                    break

            epsilon = max(0.01, epsilon * decay)
            rewards_per_episode.append(total_reward)

        # Lưu Q-table
        with open("improved_q_table.pkl", "wb") as f:
            pickle.dump(q_table, f)

        # Trích xuất đường đi từ Q-table
        path = [initial_state]
        state = initial_state
        visited = set([state])
        for _ in range(max_steps):
            if state == goal_state:
                return path, total_states_explored
            if state not in q_table:
                break
            action = np.argmax(q_table[state])
            dx, dy = directions[action]
            i, j = find_blank(state)
            ni, nj = i + dx, j + dy
            if not (0 <= ni < 3 and 0 <= nj < 3):
                break
            new_state = [list(row) for row in state]
            new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
            next_state = tuple(tuple(row) for row in new_state)
            if next_state in visited:
                break
            path.append(next_state)
            visited.add(next_state)
            state = next_state

        if state == goal_state:
            return path, total_states_explored
        return None, total_states_explored

    
    def heuristic(self, state):
        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] and state[i][j] != self.goal[i][j]:
                    value = state[i][j]
                    goal_i, goal_j = divmod(value - 1, 3)
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance

    def is_solvable(self, state=None):
        if state is None:
            state = self.initial
        state_list = [num for row in state for num in row if num != 0]
        inversions = 0
        for i in range(len(state_list)):
            for j in range(i + 1, len(state_list)):
                if state_list[i] > state_list[j]:
                    inversions += 1
        return inversions % 2 == 0

    def backtracking_search(self, depth_limit=9):
        visited = set()  # Store visited states to avoid cycles
        explored_states = []  # Store all explored states
        path = []  # Store the path from empty to goal

        def is_valid_assignment(state, pos, value):
            i, j = pos
            # Ràng buộc: Ô (0,0) phải là 1
            if i == 0 and j == 0 and value != 1:
                return False

            # Ràng buộc: Mỗi số chỉ xuất hiện một lần
            for r in range(3):
                for c in range(3):
                    if (r, c) != pos and state[r][c] == value:
                        return False

            # Chỉ kiểm tra các ô liền kề
            # Ràng buộc theo hàng: ô(i,j+1) = ô(i,j) + 1 (trừ ô trống)
            if j > 0 and state[i][j - 1] is not None and value != 0 and state[i][j - 1] != value - 1:
                return False
            if j < 2 and state[i][j + 1] is not None and value != 0 and state[i][j + 1] != value + 1:
                return False

            # Ràng buộc theo cột: ô(i+1,j) = ô(i,j) + 3 (trừ ô trống)
            if i > 0 and state[i - 1][j] is not None and value != 0 and state[i - 1][j] != value - 3:
                return False
            if i < 2 and state[i + 1][j] is not None and value != 0 and state[i + 1][j] != value + 3:
                return False

            return True

        def is_solvable(state):
            flat = [state[i][j] for i in range(3) for j in range(3) if state[i][j] is not None and state[i][j] != 0]
            inversions = 0
            for i in range(len(flat)):
                for j in range(i + 1, len(flat)):
                    if flat[i] > flat[j]:
                        inversions += 1
            return inversions % 2 == 0

        def backtrack(state, assigned, pos_index):
            # Base case: All cells assigned
            if pos_index == 9:
                state_tuple = tuple(tuple(row) for row in state)
                if state_tuple == self.goal and is_solvable(state):
                    path.append(state_tuple)
                    return path
                return None

            # Get the next cell position
            i, j = divmod(pos_index, 3)
            if i >= 3 or j >= 3:
                return None

            # Create a state tuple for checking visited states
            state_tuple = tuple(tuple(row if row is not None else (None, None, None)) for row in state)
            if state_tuple in visited:
                return None
            visited.add(state_tuple)
            explored_states.append(state_tuple)

            # Try assigning each possible value
            for value in range(9):
                if value not in assigned and is_valid_assignment(state, (i, j), value):
                    # Assign the value
                    new_state = [row[:] for row in state]
                    new_state[i][j] = value
                    new_assigned = assigned | {value}

                    # Add current state to path before recursion
                    path.append(state_tuple)

                    # Recurse to the next cell
                    result = backtrack(new_state, new_assigned, pos_index + 1)
                    if result is not None:
                        return result

                    # Backtrack: Remove the state from path
                    path.pop()

            return None

        # Initialize empty matrix and start backtracking
        empty_state = [[None for _ in range(3)] for _ in range(3)]
        result = backtrack(empty_state, set(), 0)
        return result, explored_states

    def forward_checking_search(self, depth_limit=9):
        visited = set()  # Lưu các trạng thái đã thăm
        explored_states = []  # Lưu các trạng thái đã khám phá
        path = []  # Lưu đường đi từ rỗng đến mục tiêu

        def get_domain(state, pos, assigned):
            domain = []
            for value in range(9):
                if value not in assigned and self.is_valid_assignment(state, pos, value):
                    domain.append(value)
            return domain

        def forward_check(state, pos, value, domains, assigned):
            i, j = pos
            new_domains = {k: v[:] for k, v in domains.items()}
            used_values = set(state[r][c] for r in range(3) for c in range(3) if state[r][c] is not None)

            # Chỉ kiểm tra các ô liền kề
            related_positions = []
            if j > 0: related_positions.append((i, j - 1))
            if j < 2: related_positions.append((i, j + 1))
            if i > 0: related_positions.append((i - 1, j))
            if i < 2: related_positions.append((i + 1, j))

            for other_pos in related_positions:
                if other_pos not in assigned:
                    r, c = other_pos
                    new_domain = [val for val in new_domains[other_pos] if val not in used_values]
                    if (i, j) == (0, 0) and value == 1:
                        if other_pos == (0, 1):
                            new_domain = [2]
                        elif other_pos == (1, 0):
                            new_domain = [4]
                    elif value != 0:
                        if c > 0 and state[r][c - 1] is not None and state[r][c - 1] != 0:
                            new_domain = [val for val in new_domain if val == 0 or state[r][c - 1] == val - 1]
                        if c < 2 and state[r][c + 1] is not None and state[r][c + 1] != 0:
                            new_domain = [val for val in new_domain if val == 0 or state[r][c + 1] == val + 1]
                        if r > 0 and state[r - 1][c] is not None and state[r - 1][c] != 0:
                            new_domain = [val for val in new_domain if val == 0 or state[r - 1][c] == val - 3]
                        if r < 2 and state[r + 1][c] is not None and state[r + 1][c] != 0:
                            new_domain = [val for val in new_domain if val == 0 or state[r + 1][c] == val + 3]
                    new_domains[other_pos] = new_domain
                    if not new_domain:
                        return False, domains
            return True, new_domains

        def select_mrv_variable(positions, domains, state):
            min_domain_size = float('inf')
            selected_pos = None
            for pos in positions:
                domain_size = len(domains[pos])
                if domain_size < min_domain_size:
                    min_domain_size = domain_size
                    selected_pos = pos
            return selected_pos

        def select_lcv_value(pos, domain, state, domains, assigned):
            value_scores = []
            for value in domain:
                temp_state = [row[:] for row in state]
                temp_state[pos[0]][pos[1]] = value
                _, new_domains = forward_check(temp_state, pos, value, domains, assigned)
                eliminated = sum(len(domains[p]) - len(new_domains[p]) for p in new_domains if p != pos)
                value_scores.append((eliminated, value))
            value_scores.sort()
            return [value for _, value in value_scores]

        def backtrack_with_fc(state, assigned, positions, domains):
            if len(assigned) == 9:  # Đã gán hết 9 ô
                state_tuple = tuple(tuple(row) for row in state)
                if state_tuple == self.goal and self.is_solvable(state):
                    path.append(state_tuple)
                    return path
                return None

            # Kiểm tra sớm trạng thái mục tiêu khi gán từ 7 ô trở lên
            if len(assigned) >= 7:
                temp_state = [row[:] for row in state]
                temp_assigned = assigned.copy()
                temp_positions = [p for p in positions if p not in assigned]
                temp_domains = {k: v[:] for k, v in domains.items()}
                for p in temp_positions:
                    remaining_values = [v for v in range(9) if v not in temp_assigned.values()]
                    if not remaining_values:
                        return None
                    value = remaining_values[0]  # Chọn giá trị đầu tiên
                    temp_state[p[0]][p[1]] = value
                    temp_assigned[p] = value
                    temp_tuple = tuple(tuple(row) for row in temp_state)
                    path.append(temp_tuple)  # Thêm trạng thái trung gian
                    success, temp_domains = forward_check(temp_state, p, value, temp_domains, temp_assigned)
                    if not success:
                        path.pop()
                        return None
                state_tuple = tuple(tuple(row) for row in temp_state)
                if state_tuple == self.goal and self.is_solvable(temp_state):
                    return path
                path.pop(len(temp_positions))  # Xóa các trạng thái trung gian nếu thất bại
                return None

            # Chọn ô có ít giá trị hợp lệ nhất (MRV)
            pos = select_mrv_variable(positions, domains, state)
            if pos is None:
                return None

            # Lấy tập giá trị hợp lệ và sắp xếp theo LCV
            domain = get_domain(state, pos, set(assigned.values()))
            sorted_values = select_lcv_value(pos, domain, state, domains, assigned)

            # Tạo bản sao trạng thái
            state_tuple = tuple(tuple(row if row is not None else (None, None, None)) for row in state)
            if state_tuple in visited:
                return None
            visited.add(state_tuple)
            explored_states.append(state_tuple)

            # Thử gán các giá trị theo thứ tự LCV
            for value in sorted_values:
                new_state = [row[:] for row in state]
                new_state[pos[0]][pos[1]] = value
                new_assigned = assigned.copy()
                new_assigned[pos] = value
                new_positions = [p for p in positions if p != pos]
                path.append(state_tuple)  # Thêm trạng thái trước khi gán

                # Thực hiện Forward Checking
                success, new_domains = forward_check(new_state, pos, value, domains, new_assigned)
                if success:
                    result = backtrack_with_fc(new_state, new_assigned, new_positions, new_domains)
                    if result is not None:
                        return result
                path.pop()  # Quay lui: xóa trạng thái nếu không thành công

            return None

        # Khởi tạo ma trận rỗng và tập giá trị ban đầu
        empty_state = [[None for _ in range(3)] for _ in range(3)]
        positions = [(i, j) for i in range(3) for j in range(3)]
        domains = {(i, j): list(range(9)) for i in range(3) for j in range(3)}
        assigned = {}
        result = backtrack_with_fc(empty_state, assigned, positions, domains)
        return result, explored_states

    def min_conflicts_search(self, max_iterations=1000, max_no_improvement=100, timeout=5.0):
        """
        Min-Conflicts Search for CSP following the theoretical approach.
        Starts with unassigned variables, assigns initial values, and iteratively
        selects a conflicting variable to reassign with a value that minimizes conflicts.

        Args:
            max_iterations (int): Maximum number of iterations.
            max_no_improvement (int): Maximum iterations without improvement before restart.
            timeout (float): Maximum running time in seconds.

        Returns:
            tuple: (path, num_explored_states) if solution found,
                   (None, num_explored_states) otherwise.
        """

        def count_conflicts(state):
            """
            Count the number of constraint violations in the state.

            Returns:
                int: Number of conflicts.
            """
            conflicts = 0
            value_counts = defaultdict(int)

            # Constraint: (0,0) must be 1
            if state[0][0] != 1:
                conflicts += 1

            # Constraint: Each number appears exactly once
            for i in range(3):
                for j in range(3):
                    val = state[i][j]
                    value_counts[val] += 1
                    if value_counts[val] > 1:
                        conflicts += value_counts[val] - 1

            # Row constraint: state[i][j+1] = state[i][j] + 1 (except blank)
            for i in range(3):
                for j in range(2):
                    if state[i][j] != 0 and state[i][j + 1] != 0:
                        if state[i][j + 1] != state[i][j] + 1:
                            conflicts += 1

            # Column constraint: state[i+1][j] = state[i][j] + 3 (except blank)
            for j in range(3):
                for i in range(2):
                    if state[i][j] != 0 and state[i + 1][j] != 0:
                        if state[i + 1][j] != state[i][j] + 3:
                            conflicts += 1

            # Solvability constraint (only check if state is complete)
            if all(state[i][j] is not None for i in range(3) for j in range(3)):
                if not self.is_solvable(state):
                    conflicts += 1

            return conflicts

        def get_conflicting_positions(state):
            """
            Identify positions that cause conflicts.

            Returns:
                list: List of (i, j) positions with conflicts.
            """
            conflicts = []
            value_counts = defaultdict(int)
            conflict_positions = set()

            # Check (0,0) must be 1
            if state[0][0] != 1:
                conflict_positions.add((0, 0))

            # Check unique values
            for i in range(3):
                for j in range(3):
                    val = state[i][j]
                    value_counts[val] += 1
                    if value_counts[val] > 1:
                        conflict_positions.add((i, j))

            # Check row constraints
            for i in range(3):
                for j in range(2):
                    if state[i][j] != 0 and state[i][j + 1] != 0:
                        if state[i][j + 1] != state[i][j] + 1:
                            conflict_positions.add((i, j))
                            conflict_positions.add((i, j + 1))

            # Check column constraints
            for j in range(3):
                for i in range(2):
                    if state[i][j] != 0 and state[i + 1][j] != 0:
                        if state[i + 1][j] != state[i][j] + 3:
                            conflict_positions.add((i, j))
                            conflict_positions.add((i + 1, j))

            # Check solvability
            if all(state[i][j] is not None for i in range(3) for j in range(3)):
                if not self.is_solvable(state):
                    for i in range(3):
                        for j in range(3):
                            conflict_positions.add((i, j))

            return list(conflict_positions)

        def select_min_conflict_value(state, i, j, current_value, assigned_values):
            """
            Select a value for position (i, j) that minimizes conflicts, possibly by swapping.

            Args:
                state: Current state of the puzzle.
                i, j: Position to assign a value.
                current_value: Current value at (i, j).
                assigned_values: Set of values already used.

            Returns:
                tuple: (new_value, swap_pos) where new_value is the value to assign,
                       and swap_pos is the position to swap with (or None if no swap).
            """
            value_scores = []
            state_copy = [row[:] for row in state]

            # Try swapping with other positions
            for r in range(3):
                for c in range(3):
                    if (r, c) != (i, j):
                        state_copy = [row[:] for row in state]
                        state_copy[i][j], state_copy[r][c] = state_copy[r][c], state_copy[i][j]
                        conflicts = count_conflicts(state_copy)
                        value_scores.append((conflicts, state[r][c], (r, c)))

            # Try assigning new values not in assigned_values
            for value in range(9):
                if value not in assigned_values - ({current_value} if current_value is not None else set()):
                    if (i, j) == (0, 0) and value != 1:
                        continue
                    state_copy = [row[:] for row in state]
                    state_copy[i][j] = value
                    conflicts = count_conflicts(state_copy)
                    value_scores.append((conflicts, value, None))

            if not value_scores:
                return None, None

            value_scores.sort()
            return value_scores[0][1], value_scores[0][2]

        def initialize_state():
            """
            Generate a random initial assignment for all variables.

            Returns:
                list: A 3x3 matrix with a valid initial assignment.
            """
            state = [[None for _ in range(3)] for _ in range(3)]
            numbers = list(range(9))
            random.shuffle(numbers)
            state[0][0] = 1  # Enforce (0,0) = 1
            numbers.remove(1)
            idx = 0
            for i in range(3):
                for j in range(3):
                    if (i, j) != (0, 0):
                        state[i][j] = numbers[idx]
                        idx += 1
            return state

        start_time = time.time()
        current_state = initialize_state()
        path = [tuple(tuple(row) for row in current_state)]
        num_explored_states = 1
        best_conflicts = float('inf')
        best_state = [row[:] for row in current_state]
        no_improvement_count = 0
        assigned_values = set(range(9))
        assigned_positions = {(i, j) for i in range(3) for j in range(3)}

        for iteration in range(max_iterations):
            if time.time() - start_time > timeout:
                print("Timeout reached")
                return None, num_explored_states

            current_state_tuple = tuple(tuple(row) for row in current_state)
            conflicts = count_conflicts(current_state)

            # Check if current state is a solution
            if current_state_tuple == self.goal and self.is_solvable(current_state):
                print(f"Solution found after {iteration} iterations")
                return path, num_explored_states

            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_state = [row[:] for row in current_state]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Restart with a new random assignment if no improvement
            if no_improvement_count >= max_no_improvement:
                current_state = initialize_state()
                assigned_values = set(range(9))
                assigned_positions = {(i, j) for i in range(3) for j in range(3)}
                current_state_tuple = tuple(tuple(row) for row in current_state)
                path.append(current_state_tuple)
                num_explored_states += 1
                conflicts = count_conflicts(current_state)
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_state = [row[:] for row in current_state]
                no_improvement_count = 0
                continue

            # Select a conflicting position
            conflicting_positions = get_conflicting_positions(current_state)
            if not conflicting_positions:
                if conflicts == 0 and self.is_solvable(current_state):
                    print(f"Solution found after {iteration} iterations")
                    return path, num_explored_states
                else:
                    # No conflicts but not a solution, restart
                    current_state = initialize_state()
                    assigned_values = set(range(9))
                    assigned_positions = {(i, j) for i in range(3) for j in range(3)}
                    current_state_tuple = tuple(tuple(row) for row in current_state)
                    path.append(current_state_tuple)
                    num_explored_states += 1
                    continue

            # Randomly select a conflicting position
            i, j = random.choice(conflicting_positions)
            current_value = current_state[i][j]

            # Select a value (or swap) that minimizes conflicts
            new_value, swap_pos = select_min_conflict_value(current_state, i, j, current_value, assigned_values)

            if new_value is None:
                continue

            # Update state
            current_state_list = [row[:] for row in current_state]
            if swap_pos:
                r, c = swap_pos
                current_state_list[i][j], current_state_list[r][c] = current_state_list[r][c], current_state_list[i][j]
            else:
                current_state_list[i][j] = new_value
                assigned_values.remove(current_value)
                assigned_values.add(new_value)

            current_state = current_state_list
            current_state_tuple = tuple(tuple(row) for row in current_state)
            path.append(current_state_tuple)
            num_explored_states += 1

        # Check if the best state is a solution
        if tuple(tuple(row) for row in best_state) == self.goal and self.is_solvable(best_state):
            print("Returning best state as solution")
            return path, num_explored_states
        print("No solution found")
        return None, num_explored_states

# Hàm chọn trạng thái ban đầu bằng Pygame
def initial_state_selector(goal_state):
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("8-Puzzle Initial State Selector")

    try:
        background = pygame.image.load("Image/background.jpg")
        background = pygame.transform.scale(background, (WIDTH, HEIGHT))
    except:
        background = pygame.Surface((WIDTH, HEIGHT))
        background.fill((255, 255, 255))

    title_font = pygame.font.SysFont("Arial", 50, bold=True)
    label_font = pygame.font.SysFont("Arial", 40, bold=True)
    error_font = pygame.font.SysFont("Arial", 30, bold=True)
    input_font = pygame.font.SysFont("Arial", 25, bold=True)

    initial_state = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
    tile_size = 80
    grid_offset_x = 50
    grid_offset_y = 200
    goal_offset_x = 500
    goal_offset_y = 200
    selected_cell = None
    input_active = False
    input_text = ""

    button_random_rect = pygame.Rect(150, 500, 150, 50)
    button_manual_rect = pygame.Rect(320, 500, 150, 50)
    button_confirm_rect = pygame.Rect(500, 500, 150, 50)

    def draw_grid(state, offset_x, offset_y, tile_size, selected=None):
        for i in range(3):
            for j in range(3):
                rect = pygame.Rect(offset_x + j * tile_size, offset_y + i * tile_size, tile_size, tile_size)
                if selected == (i, j):
                    pygame.draw.rect(screen, (255, 215, 0), rect, border_radius=10)
                else:
                    pygame.draw.rect(screen, (135, 206, 235), rect, border_radius=10)
                pygame.draw.rect(screen, (0, 0, 0), rect, 3, border_radius=10)
                if state[i][j] != 0:
                    text = label_font.render(str(state[i][j]), True, (255, 69, 0))
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)

    def is_valid_state(state):
        flat = [num for row in state for num in row]
        return sorted(flat) == list(range(9))

    def render_text_with_border(surface, text, font, color, border_color, pos):
        text_surface = font.render(text, True, color)
        border_surface = font.render(text, True, border_color)
        text_rect = text_surface.get_rect(center=pos)

        offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in offsets:
            border_rect = text_rect.copy()
            border_rect.x += dx
            border_rect.y += dy
            surface.blit(border_surface, border_rect)
        surface.blit(text_surface, text_rect)

    def parse_input(text):
        try:
            numbers = [int(x) for x in text.split() if x.isdigit() and 0 <= int(x) <= 8]
            if len(numbers) != 9:
                return None
            return [numbers[i:i + 3] for i in range(0, 9, 3)]
        except ValueError:
            return None

    puzzle_temp = EightPuzzle(initial_state, goal_state)

    running = True
    while running:
        screen.blit(background, (0, 0))

        render_text_with_border(screen, "Set Initial State", title_font, (255, 255, 255), (0, 0, 0), (WIDTH // 2, 50))

        render_text_with_border(screen, "Start", label_font, (255, 255, 255), (0, 0, 0),
                                (grid_offset_x + 120, grid_offset_y - 40))
        draw_grid(initial_state, grid_offset_x, grid_offset_y, tile_size, selected_cell)

        render_text_with_border(screen, "Goal", label_font, (255, 255, 255), (0, 0, 0),
                                (goal_offset_x + 120, goal_offset_y - 40))
        draw_grid(goal_state, goal_offset_x, goal_offset_y, tile_size)

        pygame.draw.rect(screen, (50, 205, 50), button_random_rect, border_radius=10)
        render_text_with_border(screen, "Random", label_font, (255, 255, 255), (0, 0, 0), button_random_rect.center)

        pygame.draw.rect(screen, (50, 205, 50), button_manual_rect, border_radius=10)
        render_text_with_border(screen, "Manual", label_font, (255, 255, 255), (0, 0, 0), button_manual_rect.center)

        pygame.draw.rect(screen, (50, 205, 50), button_confirm_rect, border_radius=10)
        render_text_with_border(screen, "Confirm", label_font, (255, 255, 255), (0, 0, 0), button_confirm_rect.center)

        if input_active:
            input_box = pygame.Rect(150, 550, 500, 40)
            pygame.draw.rect(screen, (255, 255, 255), input_box, border_radius=10)
            pygame.draw.rect(screen, (0, 0, 0), input_box, 2, border_radius=10)
            text_surface = input_font.render(input_text, True, (0, 0, 0))
            screen.blit(text_surface, (input_box.x + 5, input_box.y + 5))
            render_text_with_border(screen, "Enter 9 numbers (0-8) separated by spaces", input_font, (255, 0, 0), (0, 0, 0),
                                    (WIDTH // 2, 595))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return None  # Thoát chương trình mà không gọi pygame.quit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                for i in range(3):
                    for j in range(3):
                        rect = pygame.Rect(grid_offset_x + j * tile_size, grid_offset_y + i * tile_size, tile_size,
                                          tile_size)
                        if rect.collidepoint(mouse_pos):
                            selected_cell = (i, j)
                            break
                if button_random_rect.collidepoint(mouse_pos):
                    initial_state = list(map(list, puzzle_temp.generate_random_state()))
                    selected_cell = None
                    input_active = False
                    input_text = ""
                if button_manual_rect.collidepoint(mouse_pos):
                    input_active = True
                    selected_cell = None
                if button_confirm_rect.collidepoint(mouse_pos):
                    if input_active:
                        new_state = parse_input(input_text)
                        if new_state and is_valid_state(new_state) and puzzle_temp.is_solvable(new_state):
                            initial_state = new_state
                            input_active = False
                            input_text = ""
                        else:
                            render_text_with_border(screen, "Invalid Input!", error_font, (255, 0, 0), (0, 0, 0),
                                                    (WIDTH // 2, 450))
                            pygame.display.flip()
                            pygame.time.delay(1000)
                    else:
                        if is_valid_state(initial_state) and puzzle_temp.is_solvable(initial_state):
                            running = False  # Thoát vòng lặp mà không gọi pygame.quit()
                            return initial_state
                        else:
                            render_text_with_border(screen, "Invalid State!", error_font, (255, 0, 0), (0, 0, 0),
                                                    (WIDTH // 2, 450))
                            pygame.display.flip()
                            pygame.time.delay(1000)

            elif event.type == pygame.KEYDOWN and selected_cell:
                i, j = selected_cell
                if event.unicode.isdigit() and 0 <= int(event.unicode) <= 8:
                    initial_state[i][j] = int(event.unicode)
                elif event.key == pygame.K_BACKSPACE:
                    initial_state[i][j] = 0
            elif event.type == pygame.KEYDOWN and input_active:
                if event.key == pygame.K_RETURN:
                    new_state = parse_input(input_text)
                    if new_state and is_valid_state(new_state) and puzzle_temp.is_solvable(new_state):
                        initial_state = new_state
                        input_active = False
                        input_text = ""
                    else:
                        render_text_with_border(screen, "Invalid Input!", error_font, (255, 0, 0), (0, 0, 0),
                                                (WIDTH // 2, 450))
                        pygame.display.flip()
                        pygame.time.delay(1000)
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.unicode.isspace() or (event.unicode.isdigit() and len(input_text.split()) < 9):
                    input_text += event.unicode

    return initial_state

def draw_performance(screen, performance_history, WIDTH, HEIGHT):
    screen.fill((255, 255, 255))
    title_font = pygame.font.SysFont("Arial", 40, bold=True)
    label_font = pygame.font.SysFont("Arial", 20, bold=True)
    info_font = pygame.font.SysFont("Arial", 18, bold=True)

    algorithms = []
    avg_states_explored = []
    avg_runtimes = []

    for algo, runs in performance_history.items():
        if runs:
            algorithms.append(algo)
            states = [run["states_explored"] if isinstance(run["states_explored"], (int, float)) else len(run["states_explored"]) for run in runs]
            avg_states_explored.append(sum(states) / len(states))
            runtimes = [run["runtime"] for run in runs]
            avg_runtimes.append(sum(runtimes) / len(runtimes))

    if not algorithms:
        title = title_font.render("No Data Available", True, (0, 0, 0))
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))
        pygame.display.flip()
        return

    chart_width = WIDTH - 100
    chart_height = (HEIGHT - 200) // 2
    bar_width = chart_width // (len(algorithms) * 2)
    max_states = max(avg_states_explored)
    max_runtime = max(avg_runtimes)

    show_info_rect = pygame.Rect(WIDTH - 160, 20, 140, 40)
    pygame.draw.rect(screen, (50, 205, 50), show_info_rect, border_radius=10)
    pygame.draw.rect(screen, (0, 0, 0), show_info_rect, 2, border_radius=10)
    screen.blit(label_font.render("Show Info", True, (255, 255, 255)), show_info_rect.move(20, 5))

    def draw_chart(title, y_offset, values, max_value, color, ylabel):
        screen.blit(title_font.render(title, True, (0, 0, 0)), (WIDTH // 2 - 100, y_offset - 40))
        for i, (algo, value) in enumerate(zip(algorithms, values)):
            height = (value / max_value) * chart_height
            bar_rect = pygame.Rect(50 + i * (bar_width + 20), y_offset + chart_height - height, bar_width, height)
            pygame.draw.rect(screen, color, bar_rect)
            pygame.draw.rect(screen, (0, 0, 0), bar_rect, 2)
            screen.blit(label_font.render(f"{value:.1f}", True, (0, 0, 0)), (bar_rect.centerx - 10, bar_rect.y - 25))
            screen.blit(label_font.render(algo, True, (0, 0, 0)), (bar_rect.centerx - 20, y_offset + chart_height + 10))
        screen.blit(label_font.render(ylabel, True, (0, 0, 0)), (10, y_offset + chart_height // 2 - 10))

    draw_chart("States Explored", 80, avg_states_explored, max_states, (135, 206, 235), "States")
    draw_chart("Runtime (ms)", HEIGHT // 2 + 20, avg_runtimes, max_runtime, (255, 99, 71), "Milliseconds")
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if show_info_rect.collidepoint(event.pos):
                    try:
                        os.system("notepad algorithm_info.txt")
                    except:
                        print("File algorithm_info.txt not found.")
                else:
                    waiting = False
            elif event.type == pygame.KEYDOWN:
                waiting = False


def save_algorithm_info(algo_name, runtime, steps, states_explored, path_length):
    with open("algorithm_info.txt", "a") as f:
        f.write(f"\n=== Algorithm Performance Info ===\n")
        f.write(f"Algorithm: {algo_name}\n")
        f.write(f"Run time: {runtime:.2f} ms\n")
        f.write(f"Steps: {steps}\n")
        f.write(f"States Explored: {states_explored}\n")
        f.write(f"Path Length: {path_length}\n")
        f.write("-" * 30 + "\n")

def draw_tooltip_multiline(surface, text, font, box_rect, text_color=(255, 255, 255), bg_color=(0, 0, 0, 160)):
    # Vẽ nền mờ
    tooltip_surface = pygame.Surface((box_rect.width, box_rect.height), pygame.SRCALPHA)
    tooltip_surface.fill(bg_color)
    surface.blit(tooltip_surface, (box_rect.x, box_rect.y))

    # Vẽ từng dòng
    lines = []
    words = text.split()
    line = ""
    for word in words:
        if font.size(line + " " + word)[0] <= box_rect.width - 10:
            line += " " + word
        else:
            lines.append(line.strip())
            line = word
    lines.append(line.strip())

    y = box_rect.y + 5
    for line in lines:
        rendered = font.render(line, True, text_color)
        surface.blit(rendered, (box_rect.x + 5, y))
        y += rendered.get_height() + 2


def main_game(initial_state, goal_state):
    tooltip_font = pygame.font.SysFont("Arial", 18, bold=False)
    tooltip_texts = {
    "BFS": "BFS: Duyệt theo chiều rộng, tìm đường đi ngắn nhất.",
    "DFS": "DFS: Duyệt theo chiều sâu, có thể kẹt nhánh.",
    "UCS": "UCS: Tìm đường đi tối ưu theo chi phí.",
    "IDS": "IDS: Kết hợp BFS và DFS theo độ sâu tăng dần.",
    "Greedy": "Greedy: Dựa trên heuristic, nhanh nhưng không tối ưu.",
    "A*": "A*: f(n) = g + h, heuristic dẫn đường tốt.",
    "IDA*": "IDA*: Phiên bản tối ưu hóa bộ nhớ của A*.",
    "SimpleHC": "Simple HC: Leo đồi đơn giản, dễ kẹt cực trị.",
    "SteepHC": "Steepest HC: Chọn nước đi cải thiện tốt nhất.",
    "RandomHC": "Random HC: Leo đồi chọn ngẫu nhiên.",
    "SA": "SA: Làm nguội mô phỏng, tránh kẹt cực trị.",
    "Beam": "Beam: Tìm kiếm theo chùm k trạng thái tốt nhất.",
    "Genetic": "Genetic: Tiến hóa qua lai ghép, đột biến.",
    "AND-OR": "AND-OR: Tìm chiến lược có điều kiện trong thế giới không chắc chắn.",
    "Belief": "Belief: Làm việc với nhiều trạng thái niềm tin.",
    "PartObs": "POS: Tìm kiếm trong môi trường quan sát một phần.",
    "Backtrack": "Backtrack: Gán giá trị từng bước, quay lui khi cần.",
    "Forward": "Forward Checking: Loại bỏ giá trị sai trước khi gán.",
    "MinConf": "Min-Conflicts: Sửa dần trạng thái để giảm xung đột.",
    "QLearn": "Q-Learning: Học chính sách từ thử sai."
}

    WIDTH, HEIGHT = 1200, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("8-Puzzle Solver")

    try:
        background = pygame.image.load("Image/background.jpg")
        background = pygame.transform.scale(background, (WIDTH, HEIGHT))
    except:
        background = pygame.Surface((WIDTH, HEIGHT))
        background.fill((255, 255, 255))

    title_font = pygame.font.SysFont("Arial", 60, bold=True)
    label_font = pygame.font.SysFont("Arial", 30, bold=True)
    button_font = pygame.font.SysFont("Arial", 20, bold=True)
    info_font = pygame.font.SysFont("Arial", 25, bold=True)
    number_font = pygame.font.SysFont("Arial", 50, bold=True)

    puzzle = EightPuzzle(initial_state, goal_state)

    small_tile_size = 60
    initial_grid_x, initial_grid_y = 50, 150
    goal_grid_x, goal_grid_y = 250, 150
    algo_tile_size = 120
    algo_grid_x, algo_grid_y = 450, 70

    button_width, button_height = 120, 40
    button_spacing_x, button_spacing_y = 130, 50
    start_x, start_y = 360, 460
    buttons = [
        ("BFS", pygame.Rect(start_x, start_y, button_width, button_height)),
        ("DFS", pygame.Rect(start_x + button_spacing_x, start_y, button_width, button_height)),
        ("UCS", pygame.Rect(start_x + 2 * button_spacing_x, start_y, button_width, button_height)),
        ("IDS", pygame.Rect(start_x + 3 * button_spacing_x, start_y, button_width, button_height)),
        ("Greedy", pygame.Rect(start_x, start_y + button_spacing_y, button_width, button_height)),
        ("A*", pygame.Rect(start_x + button_spacing_x, start_y + button_spacing_y, button_width, button_height)),
        ("IDA*", pygame.Rect(start_x + 2 * button_spacing_x, start_y + button_spacing_y, button_width, button_height)),
        ("SimpleHC", pygame.Rect(start_x + 3 * button_spacing_x, start_y + button_spacing_y, button_width, button_height)),
        ("SteepHC", pygame.Rect(start_x, start_y + 2 * button_spacing_y, button_width, button_height)),
        ("RandomHC", pygame.Rect(start_x + button_spacing_x, start_y + 2 * button_spacing_y, button_width, button_height)),
        ("SA", pygame.Rect(start_x + 2 * button_spacing_x, start_y + 2 * button_spacing_y, button_width, button_height)),
        ("Beam", pygame.Rect(start_x + 3 * button_spacing_x, start_y + 2 * button_spacing_y, button_width, button_height)),
        ("Genetic", pygame.Rect(start_x, start_y + 3 * button_spacing_y, button_width, button_height)),
        ("AND-OR", pygame.Rect(start_x + button_spacing_x, start_y + 3 * button_spacing_y, button_width, button_height)),
        ("Belief", pygame.Rect(start_x + 2 * button_spacing_x, start_y + 3 * button_spacing_y, button_width, button_height)),
        ("PartObs", pygame.Rect(start_x + 3 * button_spacing_x, start_y + 3 * button_spacing_y, button_width, button_height)),
        ("Backtrack", pygame.Rect(start_x, start_y + 4 * button_spacing_y, button_width, button_height)),
        ("Forward", pygame.Rect(start_x + button_spacing_x, start_y + 4 * button_spacing_y, button_width, button_height)),
        ("MinConf", pygame.Rect(start_x + 2 * button_spacing_x, start_y + 4 * button_spacing_y, button_width, button_height)),
        ("QLearn", pygame.Rect(start_x + 3 * button_spacing_x, start_y + 4 * button_spacing_y, button_width, button_height)),
    ]

    back_button_rect = pygame.Rect(WIDTH - 150, HEIGHT - 50, 120, 40)
    reset_button_rect = pygame.Rect(WIDTH - 300, HEIGHT - 50, 120, 40)
    view_button_rect = pygame.Rect(WIDTH - 450, HEIGHT - 50, 120, 40)
    show_info_button_rect = pygame.Rect(WIDTH - 600, HEIGHT - 50, 120, 40)
    reset_chart_button_rect = pygame.Rect(WIDTH - 750, HEIGHT - 50, 120, 40)


    performance_history = {
        "BFS": [], "DFS": [], "UCS": [], "IDS": [], "Greedy": [], "A*": [], "IDA*": [],
        "SimpleHC": [], "SteepHC": [], "RandomHC": [], "SA": [], "Beam": [], "Genetic": [],
        "AND-OR": [], "Belief": [], "PartObs": [], "Backtrack": [], "Forward": [], "MinConf": [], "QLearn": []
    }

    solution = None
    solution_index = 0
    elapsed_time = 0
    steps = 0
    error_message = None
    error_timer = 0
    selected_button = None
    display_state = initial_state

    def draw_grid(state, offset_x, offset_y, tile_size):
        for i in range(3):
            for j in range(3):
                rect = pygame.Rect(offset_x + j * tile_size, offset_y + i * tile_size, tile_size, tile_size)
                if state[i][j] == 0:
                    pygame.draw.rect(screen, (255, 255, 255), rect, border_radius=8)
                else:
                    pygame.draw.rect(screen, (173, 216, 230), rect, border_radius=8)
                pygame.draw.rect(screen, (0, 0, 0), rect, 2, border_radius=8)
                if state[i][j] != 0:
                    num_font = number_font if tile_size > 50 else pygame.font.SysFont("Arial", 30, bold=True)
                    text = num_font.render(str(state[i][j]), True, (255, 140, 0))
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)

    def render_text_with_border(surface, text, font, color, border_color, pos):
        text_surface = font.render(text, True, color)
        border_surface = font.render(text, True, border_color)
        text_rect = text_surface.get_rect(center=pos)

        offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in offsets:
            border_rect = text_rect.copy()
            border_rect.x += dx
            border_rect.y += dy
            surface.blit(border_surface, border_rect)
        surface.blit(text_surface, text_rect)

    running = True
    current_tooltip = ""
    clock = pygame.time.Clock()
    while running:
        screen.blit(background, (0, 0))

        render_text_with_border(screen, "8-Puzzle Solver", title_font, (255, 255, 255), (0, 0, 0), (WIDTH // 2, 30))

        render_text_with_border(screen, "Initial", label_font, (255, 255, 255), (0, 0, 0),
                                (initial_grid_x + (small_tile_size * 3) // 2, initial_grid_y - 20))
        draw_grid(initial_state, initial_grid_x, initial_grid_y, small_tile_size)

        render_text_with_border(screen, "Goal", label_font, (255, 255, 255), (0, 0, 0),
                                (goal_grid_x + (small_tile_size * 3) // 2, goal_grid_y - 20))
        draw_grid(goal_state, goal_grid_x, goal_grid_y, small_tile_size)

        if solution:
            if solution_index < len(solution):
                current_state = solution[solution_index]
                if isinstance(current_state, list) and all(isinstance(s, tuple) for s in current_state):
                    for idx, sub_state in enumerate(current_state):
                        offset_x = algo_grid_x + (idx * algo_tile_size * 3)
                        draw_grid(sub_state, offset_x, algo_grid_y, algo_tile_size)
                else:
                    draw_grid(current_state, algo_grid_x, algo_grid_y, algo_tile_size)
                solution_index += 1
                pygame.time.wait(200)
            else:
                draw_grid(solution[-1], algo_grid_x, algo_grid_y, algo_tile_size)
        else:
            draw_grid(display_state, algo_grid_x, algo_grid_y, algo_tile_size)

        render_text_with_border(screen, f"Time: {elapsed_time:.2f} ms", info_font, (255, 255, 255), (0, 0, 0), (920, 280))
        render_text_with_border(screen, f"Steps: {steps}", info_font, (255, 255, 255), (0, 0, 0), (920, 320))

        if error_message and pygame.time.get_ticks() - error_timer < 1000:
            render_text_with_border(screen, "No Solution!", info_font, (255, 0, 0), (0, 0, 0),
                                    (algo_grid_x + 180, algo_grid_y + 500))

        for label, rect in buttons:
            button_color = (255, 165, 0) if selected_button == label else (255, 215, 0)
            pygame.draw.rect(screen, button_color, rect, border_radius=10)
            pygame.draw.rect(screen, (0, 0, 0), rect, 2, border_radius=10)
            render_text_with_border(screen, label, button_font, (255, 255, 255), (0, 0, 0), rect.center)

        pygame.draw.rect(screen, (255, 215, 0), view_button_rect, border_radius=10)
        pygame.draw.rect(screen, (0, 0, 0), view_button_rect, 2, border_radius=10)
        render_text_with_border(screen, "View Stats", button_font, (255, 255, 255), (0, 0, 0), view_button_rect.center)

        pygame.draw.rect(screen, (255, 215, 0), reset_button_rect, border_radius=10)
        pygame.draw.rect(screen, (0, 0, 0), reset_button_rect, 2, border_radius=10)
        render_text_with_border(screen, "Reset", button_font, (255, 255, 255), (0, 0, 0), reset_button_rect.center)

        pygame.draw.rect(screen, (255, 215, 0), back_button_rect, border_radius=10)
        pygame.draw.rect(screen, (0, 0, 0), back_button_rect, 2, border_radius=10)
        render_text_with_border(screen, "Back", button_font, (255, 255, 255), (0, 0, 0), back_button_rect.center)

        pygame.draw.rect(screen, (255, 215, 0), show_info_button_rect, border_radius=10)
        pygame.draw.rect(screen, (0, 0, 0), show_info_button_rect, 2, border_radius=10)
        render_text_with_border(screen, "Show Info", button_font, (255, 255, 255), (0, 0, 0), show_info_button_rect.center)
        
        pygame.draw.rect(screen, (255, 215, 0), reset_chart_button_rect, border_radius=10)
        pygame.draw.rect(screen, (0, 0, 0), reset_chart_button_rect, 2, border_radius=10)
        render_text_with_border(screen, "Reset Chart", button_font, (255, 255, 255), (0, 0, 0), reset_chart_button_rect.center)

        if current_tooltip:
            tooltip_box = pygame.Rect(10, HEIGHT - 70, 420, 50)
            draw_tooltip_multiline(screen, current_tooltip, tooltip_font, tooltip_box)

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
                current_tooltip = ""
                for label, rect in buttons:
                    if rect.collidepoint(mouse_pos):
                        current_tooltip = tooltip_texts.get(label, "")
                        break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if show_info_button_rect.collidepoint(mouse_pos):
                    try:
                        os.system("notepad algorithm_info.txt")
                    except:
                        print("File algorithm_info.txt not found.")
                    continue

                if view_button_rect.collidepoint(mouse_pos):
                    draw_performance_plotly(performance_history)
                    continue

                if reset_button_rect.collidepoint(mouse_pos):
                    solution = None
                    solution_index = 0
                    elapsed_time = 0
                    steps = 0
                    error_message = None
                    display_state = initial_state
                    continue

                if back_button_rect.collidepoint(mouse_pos):
                    return "BACK"

                if reset_chart_button_rect.collidepoint(mouse_pos):
                    performance_history.clear()
                    for key in ["BFS", "DFS", "UCS", "IDS", "Greedy", "A*", "IDA*", "SimpleHC", "SteepHC", "RandomHC", "SA", "Beam", "Genetic",
                                "AND-OR", "Belief", "PartObs", "Backtrack", "Forward", "MinConf", "QLearn"]:
                        performance_history[key] = []
                    continue


                for label, rect in buttons:
                    if rect.collidepoint(mouse_pos):
                        selected_button = label
                        solution = None
                        solution_index = 0
                        elapsed_time = 0
                        steps = 0
                        error_message = None
                        display_state = initial_state

                        try:  # Thêm xử lý lỗi
                            start_time = timeit.default_timer()
                            if label == "BFS":
                                solution, explored_states = puzzle.bfs()
                            elif label == "DFS":
                                solution, explored_states = puzzle.dfs()
                            elif label == "UCS":
                                solution, explored_states = puzzle.ucs()
                            elif label == "IDS":
                                solution, explored_states = puzzle.ids()
                            elif label == "Greedy":
                                solution, explored_states = puzzle.greedy()
                            elif label == "A*":
                                solution, explored_states = puzzle.a_star()
                            elif label == "IDA*":
                                solution, explored_states = puzzle.ida_star()
                            elif label == "SimpleHC":
                                solution, explored_states = puzzle.simple_hc()
                            elif label == "SteepHC":
                                solution, explored_states = puzzle.steepest_hc()
                            elif label == "RandomHC":
                                solution, explored_states = puzzle.random_hc()
                            elif label == "SA":
                                solution, explored_states = puzzle.simulated_annealing()
                            elif label == "Beam":
                                solution, explored_states = puzzle.beam_search()
                            elif label == "Genetic":
                                solution, explored_states = puzzle.genetic_algorithm()
                            elif label == "AND-OR":
                                solution, explored_states = puzzle.and_or_search()
                            elif label == "Belief":
                                result = show_belief_screen(puzzle, screen, WIDTH, HEIGHT)
                                if isinstance(result, dict):
                                    performance_history["Belief"].append(result)
                                elif result == "QUIT":
                                    return "QUIT"
                                continue

                            elif label == "PartObs":
                                result = show_pos_screen(puzzle, screen, WIDTH, HEIGHT)
                                if isinstance(result, dict):
                                    performance_history["PartObs"].append(result)
                                elif result == "QUIT":
                                    return "QUIT"
                                continue


                            elif label == "Backtrack":
                                solution, explored_states = puzzle.backtracking_search()
                            elif label == "Forward":
                                solution, explored_states = puzzle.forward_checking_search()
                            elif label == "MinConf":
                                solution, num_explored_states = puzzle.min_conflicts_search()
                                explored_states = num_explored_states  # Gán đúng số lượng trạng thái đã xét

                            elif label == "QLearn":
                                solution, explored_states = puzzle.q_learning_search()

                            elapsed_time = (timeit.default_timer() - start_time) * 1000
                            steps = len(solution) - 1 if solution else 0
                            path_length = len(solution) - 1 if solution else 0
                            states_explored = len(explored_states) if isinstance(explored_states, (list, set, tuple)) else int(explored_states)

                            if not solution:
                                error_message = True
                                error_timer = pygame.time.get_ticks()
                            performance_history[label].append({
                                "runtime": elapsed_time,
                                "steps": steps,
                                "states_explored": states_explored if isinstance(states_explored, (list, set, tuple)) else int(states_explored),
                                "path": solution if solution else []
                            })
                            save_algorithm_info(label, elapsed_time, steps, states_explored, path_length)

                        except Exception as e:
                            print(f"Error in {label} algorithm: {e}")
                            error_message = True
                            error_timer = pygame.time.get_ticks()

    return None
def show_belief_screen(puzzle, screen, WIDTH, HEIGHT):
    pygame.display.set_caption("Belief State Search")
    try:
        background = pygame.image.load("Image/background.jpg")
        background = pygame.transform.scale(background, (WIDTH, HEIGHT))
    except:
        background = pygame.Surface((WIDTH, HEIGHT))
        background.fill((255, 255, 255))


    title_font = pygame.font.SysFont("Arial", 55, bold=True)
    label_font = pygame.font.SysFont("Arial", 30, bold=True)
    number_font = pygame.font.SysFont("Arial", 45, bold=True)

    tile_size = 90
    algo_grid_x, algo_grid_y = 80, 140
    back_button = pygame.Rect(WIDTH - 150, HEIGHT - 60, 120, 40)
    run_button = pygame.Rect(WIDTH - 290, HEIGHT - 60, 120, 40)
    grid_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    initial_belief = puzzle.find_states_with_one_at_00(puzzle.initial, max_states=3)
    path = None
    explored = set()
    total_steps = 0
    elapsed_time = 0
    frame = 0

    def draw_grid(state, x, y, label):
        label_text = label_font.render(label, True, (0, 0, 0))
        screen.blit(label_text, (x + 30, y - 50))
        for i in range(3):
            for j in range(3):
                rect = pygame.Rect(x + j * tile_size, y + i * tile_size, tile_size, tile_size)
                pygame.draw.rect(grid_surface, (173, 216, 230), rect, border_radius=10)
                pygame.draw.rect(grid_surface, (0, 0, 0), rect, 2)
                if state[i][j] != 0:
                    text = number_font.render(str(state[i][j]), True, (255, 140, 0))
                    grid_surface.blit(text, text.get_rect(center=rect.center))

    clock = pygame.time.Clock()
    while True:
        screen.blit(background, (0, 0))
        grid_surface.fill((0, 0, 0, 0))

        # Hiển thị trạng thái hiện tại
        if path and frame < len(path):
            for idx, state in enumerate(path[frame]):
                draw_grid(state, algo_grid_x + idx * (tile_size * 3 + 40), algo_grid_y, f"Belief {idx + 1}")
            frame += 1
        elif path:
            for idx, state in enumerate(path[-1]):
                draw_grid(state, algo_grid_x + idx * (tile_size * 3 + 40), algo_grid_y, f"Belief {idx + 1}")
        else:
            for idx, state in enumerate(initial_belief):
                draw_grid(state, algo_grid_x + idx * (tile_size * 3 + 40), algo_grid_y, f"Belief {idx + 1}")

        screen.blit(grid_surface, (0, 0))
        title = title_font.render("Belief State Search", True, (0, 0, 0))
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 30))

        pygame.draw.rect(screen, (255, 215, 0), run_button, border_radius=8)
        pygame.draw.rect(screen, (0, 0, 0), run_button, 2)
        screen.blit(label_font.render("Run", True, (0, 0, 0)), run_button.move(35, 5))

        pygame.draw.rect(screen, (255, 215, 0), back_button, border_radius=8)
        pygame.draw.rect(screen, (0, 0, 0), back_button, 2)
        screen.blit(label_font.render("Back", True, (0, 0, 0)), back_button.move(30, 5))

        if path:
            info = label_font.render(f"Time: {elapsed_time:.2f} ms    Steps: {total_steps}", True, (0, 0, 0))
            screen.blit(info, (WIDTH - 480, HEIGHT - 110))

        pygame.display.flip()
        clock.tick(6)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if back_button.collidepoint(event.pos):
                    if path:
                        return {
                            "runtime": elapsed_time,
                            "steps": total_steps,
                            "states_explored": len(explored),
                            "path": path
                        }
                    else:
                        return None
                elif run_button.collidepoint(event.pos):
                    start_time = timeit.default_timer()
                    path, explored, total_steps = puzzle.belief_state_search(initial_belief)
                    elapsed_time = (timeit.default_timer() - start_time) * 1000
                    frame = 0



def show_pos_screen(puzzle, screen, WIDTH, HEIGHT):
    pygame.display.set_caption("Partial Observable Search")

    try:
        background = pygame.image.load("Image/background.jpg")
        background = pygame.transform.scale(background, (WIDTH, HEIGHT))
    except:
        background = pygame.Surface((WIDTH, HEIGHT))
        background.fill((255, 255, 255))


    title_font = pygame.font.SysFont("Arial", 55, bold=True)
    label_font = pygame.font.SysFont("Arial", 30, bold=True)
    number_font = pygame.font.SysFont("Arial", 45, bold=True)

    tile_size = 90
    algo_grid_x, algo_grid_y = 80, 140
    back_button = pygame.Rect(WIDTH - 150, HEIGHT - 60, 120, 40)
    run_button = pygame.Rect(WIDTH - 290, HEIGHT - 60, 120, 40)
    grid_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    initial_belief = puzzle.find_states_with_one_at_00(puzzle.initial, max_states=3)
    path = None
    explored = set()
    total_steps = 0
    elapsed_time = 0
    frame = 0

    def draw_grid(state, x, y, label):
        label_text = label_font.render(label, True, (0, 0, 0))
        screen.blit(label_text, (x + 30, y - 50))
        for i in range(3):
            for j in range(3):
                rect = pygame.Rect(x + j * tile_size, y + i * tile_size, tile_size, tile_size)
                pygame.draw.rect(grid_surface, (144, 238, 144), rect, border_radius=10)
                pygame.draw.rect(grid_surface, (0, 0, 0), rect, 2)
                if state[i][j] != 0:
                    text = number_font.render(str(state[i][j]), True, (0, 100, 0))
                    grid_surface.blit(text, text.get_rect(center=rect.center))

    clock = pygame.time.Clock()
    while True:
        screen.blit(background, (0, 0))
        grid_surface.fill((0, 0, 0, 0))

        if path and frame < len(path):
            for idx, state in enumerate(path[frame]):
                draw_grid(state, algo_grid_x + idx * (tile_size * 3 + 40), algo_grid_y, f"Belief {idx + 1}")
            frame += 1
        elif path:
            for idx, state in enumerate(path[-1]):
                draw_grid(state, algo_grid_x + idx * (tile_size * 3 + 40), algo_grid_y, f"Belief {idx + 1}")
        else:
            for idx, state in enumerate(initial_belief):
                draw_grid(state, algo_grid_x + idx * (tile_size * 3 + 40), algo_grid_y, f"Belief {idx + 1}")

        screen.blit(grid_surface, (0, 0))
        title = title_font.render("Partial Observable Search", True, (0, 0, 0))
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 30))

        pygame.draw.rect(screen, (255, 215, 0), run_button, border_radius=8)
        pygame.draw.rect(screen, (0, 0, 0), run_button, 2)
        screen.blit(label_font.render("Run", True, (0, 0, 0)), run_button.move(35, 5))

        pygame.draw.rect(screen, (255, 215, 0), back_button, border_radius=8)
        pygame.draw.rect(screen, (0, 0, 0), back_button, 2)
        screen.blit(label_font.render("Back", True, (0, 0, 0)), back_button.move(30, 5))

        if path:
            info = label_font.render(f"Time: {elapsed_time:.2f} ms    Steps: {total_steps}", True, (0, 0, 0))
            screen.blit(info, (WIDTH - 480, HEIGHT - 110))

        pygame.display.flip()
        clock.tick(6)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if back_button.collidepoint(event.pos):
                    if path:
                        return {
                            "runtime": elapsed_time,
                            "steps": total_steps,
                            "states_explored": len(explored),
                            "path": path
                        }
                    else:
                        return None
                elif run_button.collidepoint(event.pos):
                    start_time = timeit.default_timer()
                    path, explored, total_steps = puzzle.partial_observable_search()
                    elapsed_time = (timeit.default_timer() - start_time) * 1000
                    frame = 0

def render_text_with_border(surface, text, font, color, border_color, pos):
    text_surface = font.render(text, True, color)
    border_surface = font.render(text, True, border_color)
    text_rect = text_surface.get_rect(center=pos)

    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    for dx, dy in offsets:
        border_rect = text_rect.copy()
        border_rect.x += dx
        border_rect.y += dy
        surface.blit(border_surface, border_rect)

    surface.blit(text_surface, text_rect)


if __name__ == "__main__":
    pygame.init()  # Khởi tạo Pygame một lần duy nhất
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    initial_state = initial_state_selector(goal_state)

    # Kiểm tra nếu người dùng thoát trong initial_state_selector
    if initial_state is None:
        pygame.quit()
        sys.exit()

    while True:
        result = main_game(initial_state, goal_state)
        if result == "BACK":
            initial_state = initial_state_selector(goal_state)
            if initial_state is None:  # Kiểm tra nếu người dùng thoát
                break
        else:
            break

    pygame.quit()  # Chỉ gọi pygame.quit() một lần ở cuối chương trình
