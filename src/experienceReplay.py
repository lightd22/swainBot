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
        maxBufferSize (int): maximum number of experiences to store in the buffer, default value is 300.
    """
    def __init__(self, maxBufferSize = 300):
        self.buffer = []
        self.bufferSize = maxBufferSize
        self.oldestExperience = 0

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
            if len(self.buffer) < self.bufferSize:
                self.buffer.append(experience)
            else:
                self.buffer[self.oldestExperience] = experience
                self.oldestExperience += 1
                self.oldestExperience = self.oldestExperience % self.bufferSize
        return None

    def sample(self, sampleSize):
        """
        ExperienceBuffer.sample samples the current buffer using random.sample to return a collection of sampleSize experiences from the replay buffer.
        random.sample samples without replacement, so sampleSize must be no larger than the length of the current buffer.

        Args:
            sampleSize (int): number of samples to take from buffer
        Returns:
            sample (list(tuples)): list of experience replay samples. len(sample) = sampleSize.
        """
        return random.sample(self.buffer,sampleSize)

