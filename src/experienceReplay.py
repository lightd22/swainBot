import random

class ExperienceBuffer():
    """
    ExperienceBuffer is a class for storing and adding experiences to be sampled from when batch learning a Qnetwork. An experience is defined as a tuple of the form
    (s, a, r, s') where
        s  = input state
        a  = action taken from state s
        r  = reward obtained for taking action a
        s' = ending state after taking action a
    Args:
        max_buffer_size (int): maximum number of experiences to store in the buffer, default value is 300.
    """
    def __init__(self, max_buffer_size = 300):
        self.buffer = []
        self.buffer_size = max_buffer_size
        self.oldest_experience = 0

    def store(self, experiences):
        """
        ExperienceBuffer.store stores the input list of experience tuples into the buffer. The expereince is stored in one of two ways:
        1) If the buffer has space remaining, the experience is appended to the end
        2) If the buffer is full, the input experience replaces the oldest experience in the buffer

        Args:
            experiences ( list(tuple) ): each experience is a tuple of the form (s, a, r, s')
        Returns:
            None
        """
        for experience in experiences:
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(experience)
            else:
                self.buffer[self.oldest_experience] = experience
                self.oldest_experience += 1
                self.oldest_experience = self.oldest_experience % self.buffer_size
        return None

    def sample(self, sample_size):
        """
        ExperienceBuffer.sample samples the current buffer using random.sample to return a collection of sample_size experiences from the replay buffer.
        random.sample samples without replacement, so sample_size must be no larger than the length of the current buffer.

        Args:
            sample_size (int): number of samples to take from buffer
        Returns:
            sample (list(tuples)): list of experience replay samples. len(sample) = sample_size.
        """
        return random.sample(self.buffer,sample_size)

    def getBufferSize(self):
        """
        Returns length of the buffer.
        """
        return len(self.buffer)
