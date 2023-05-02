from GraphState import GraphState


class ReplayBuffer:
    def __init__(self, max_iterations, size=25, invalidate_duration=10):
        self.max_iterations = max_iterations
        self.buffer = [GraphState() for _ in range(size)]
        self.invalidate_duration = invalidate_duration

        self.num_iterations = 0
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_iterations > self.max_iterations:
            raise StopIteration
        else:
            old_iterations = self.num_iterations
            self.num_iterations += 1

            if old_iterations % self.invalidate_duration == 0:
                self.buffer = [GraphState() for _ in range(self.size)]

            return self.buffer[old_iterations - 1 % len(self.buffer)]
