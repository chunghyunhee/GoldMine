class Particle:
    def __init__(self, **kwargs):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity
        self.pos_best_i = []  # best position individual
        self.err_best_i = []  # best error individual
        self.err_i = -1       # error individual
        # inheritance init
        super(ParticleSwarmOptimization, self).__init__(**kwargs)
        self._check_hpo_params()
        self.DUP_CHECK = False


    def _check_hpo_params(self):
        self._n_pop = self._n_params
        self._n_steps = self._hpo_params["n_steps"]
        self._position_list = self._hpo_params["position_list"]               # for position
        self._bounds = self._hpo_params(["bounds"])


        for i in range(0, num_dimentions):
            # init the particle position and velocity
            self.velocity_i.append(uniform(-1, 1))
            self.position_i.append(self._position_list[i])

    ## PSO overall process
    # generate candidate function
    def _generate(self, param_list, score_list):
        result_param_list = list()
        # generate random hyperparameter
        best_param_list = self._population(param_list)
        evaluate_value = self.evaluate(param_list)
        update_velocity_params = self.update_velocity(param_list)
        update_position_params = self.update_position(param_list)

        result_param_list += evaluate_value+update_velocity_params+update_position_params
        num_result_params = len(result_param_list)

        ## leak
        if  num_result_params < self._n_pop:
            result_param_list += self._generate_param_dict_list(self._n_pop - num_result_params)
        ## over
        elif num_result_params > self._n_pop :
            random.shuffle(result_param_list)
            result_param_list = result_param_list[:self._n_pop]

        return best_param_list

    # evaluate current fitness
    def evaluate(self, param_list):
        self.err_i = self.optimize()

        # check to see if the current position is an individual best value
        if self.err_i < self.err_best_i or self.err_best_i == -1 :
            self.pos_best_i = self.pos_best_i.copy()
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self, param_list):
        w = 0.5  # constant of the inertia weight
        c1 = 1   # cognitive constants
        c2 = 2   # social constants

        for i in range(0, num_dimentions):
            r1 = random()
            r2 = random()

            vel_cognitive = c1*r1*(self.pos_best_i[i] - self.position_i[i])
            vel_social = c2*r2*(param_list[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, param_list):
        for i in range(0, num_dimentions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust max position if necessary
            if self.position_i[i] > self.bounds[i][1]:
                self.position_i[i] = self.bounds[i][1]
            # adjust min position if necessary
            if self.position_i[i] < self.bounds[i][0]:
                self.position_i[i] = self.bounds[i][0]





