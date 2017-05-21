
class Driver():

    def get_policy(self, state_batch):
        """
        gets an action for each state in batch
        :param state_batch: list of states
        :return: list of actions
        """
        raise NotImplemented

    def get_value(self, state_batch):
        """
        gets the reward for the list of given list of states
        :param state_batch: list of states to evaluate
        :return: list of floats
        """
        return [state.reward() for state in state_batch]

    def train(self, R, states):
        """
        trains the model, minimizing the error of the models output with the given states and R
        :param R: actual reward for each state in states
        :param states: list of states, corresponding to the rewards in R
        :return:
        """
        raise NotImplementedError
