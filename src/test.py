class ReptileModel:
    def __init__(self, model):
        self.model = model

    def inner_update(self, task_data, lr_inner=0.01):
        # Perform inner updates directly on the model parameters
        loss = forward_and_backward(self.model, task_data)

        # Adjust model parameters in the direction of the gradient
        for param in self.model.parameters():
            param.data.sub_(lr_inner * param.grad.data)

        return loss

    def train(self, tasks, num_outer_updates=100, lr_outer=0.001, lr_inner=0.01):
        for _ in range(num_outer_updates):
            # Randomly sample a task from the tasks
            task = random.choice(tasks)

            # Split the task into support and query sets
            support_data, query_data = split_task(task)

            # Clone the model for this iteration
            model_copy = copy.deepcopy(self.model)

            # Perform Reptile inner update
            loss = model_copy.inner_update(support_data, lr_inner)

            # Adjust the base model's parameters towards the updated parameters
            for param, param_copy in zip(self.model.parameters(), model_copy.parameters()):
                param.data.sub_(lr_outer * (param.data - param_copy.data))

            # Optionally, you can perform additional updates based on the query_data

        # Final model after num_outer_updates iterations
        return self.model