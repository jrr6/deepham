from GraphState import GraphState


class ReplayBuffer:
    def __init__(self, max_iterations, size=20, invalidate_duration=4):
        self.max_iterations = max_iterations
        self.buffer = [GraphState(regenerate_graphs=False) for _ in range(size)]
        self.invalidate_duration = invalidate_duration

        self.num_iterations = 0
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_iterations > self.max_iterations:
            raise StopIteration
        else:
            self.num_iterations += 1

            if self.num_iterations % (self.size * self.invalidate_duration) == 0:
                self.buffer = [GraphState(regenerate_graphs=False) for _ in range(self.size)]

            return self.buffer[(self.num_iterations - 1) % len(self.buffer)]
