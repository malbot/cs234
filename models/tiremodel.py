import math


class TireModel():
    """
    Full tire model, uses down force on the tire, plus the effective cornering stiffness (alpha) and longitudinal
    slip (sigma) to find the lateral/longitudinal forces on the tire (relative to the tires frame)
    """

    def __init__(self, mu_s, mu_p, cx, cy):
        """
        Creates a tire model
        :param mu_s:
        :param mu_p:
        :param cx: longitudinal tire stiffness
        :param cy: lateral tire stiffness
        """
        self.mu_s = mu_s
        self.mu_p = mu_p
        self.cx = cx
        self.cy = cy

    def __call__(self, Fz, sigma, alpha, as_dict=False):
        """
        find the Fx and Fy forces of the road on the wheel
        :param Fz: downward force
        :param sigma: longitudinal slip ratio
        :param alpha: effective cornering stiffness
        :param as_dict: returns a dictionary instead of list
        :return: the Fx and Fy of the wheel
        if as_dict==True, returns {'Fx':Fx, 'Fy':Fy}
        """
        f = math.sqrt(self.cx ** 2 * (sigma / (1 + sigma)) ** 2 + self.cy**2 * (math.tan(alpha) / (1 + sigma)) ** 2)
        t = 3*self.mu_p*Fz
        if f < t:
            F = f - 1/t * (2-self.mu_s/self.mu_p)*(f**2) + 1/(t**2)*(1-(2*self.mu_s)/(3*self.mu_p))*(f**3)
        else:
            F = self.mu_s*Fz
        if f == 0:
            Fx = 0
            Fy = 0
        else:
            Fx = self.cx*(sigma/(1+sigma))*F/f
            Fy = -1*self.cy*(math.tan(alpha)/(1+sigma))*F/f

        if as_dict:
            return {
                'Fx': Fx,
                'Fy': Fy
            }
        else:
            return Fx, Fy

def tyre_model_test():
    """
    test tire model against a validated model
    throws assertion error if model values deviate from validated values
    :return: 
    """
    max_e = 0.001  # fraction maximum error between TireModel and correct values
    Cx = 200000
    Cy = 100000
    Mu_p = 1.1
    Mu_s = 0.9
    m = 1650  # mass of the car
    mdf = .57  # mass distribution of the front
    g = 9.81  # gravity
    fz_front = m * mdf / 2 * g
    fz_rear = m*(1-mdf)/2*g

    model = TireModel(mu_s=Mu_s, mu_p=Mu_p, cx=Cx, cy=Cy)
    inputs = [
        [0.1421e-15, 0, fz_front],
        [-0.1421e-15, 0, fz_front],
        [8.0036e-04, -0.0458, fz_front],
        [6.2236e-04, -0.0464, fz_front],
        [4.0686e-04, -0.0428, fz_front],
        [3.7968e4, -1.5704, fz_rear],
        [-64.3284, 3.1411, fz_rear],
        [-5.8522e4, -1.5710, fz_rear]
    ]
    results = [
        [0.284e-10, 0],
        [-0.2842e-10, 0],
        [0.1097e3, 3.1376e3],
        [0.0848e3, 3.1649e3],
        [0.0572e3, 3.0125e3],
        [3.1304e3, 104.0143],
        [3.1321e3, -0.0119],
        [3.1293e3, 131.2711]
    ]
    for input, result in zip(inputs, results):
        output = model(input[2], input[0], input[1])
        for o, r, type in zip(output, result, ['fx', 'fy']):
            type = "{type} = {actual}, should be {correct} is incorrect for sigma {sigma} and alpha {alpha}".format(
                type=type,
                sigma=input[0],
                alpha=input[1],
                actual=o,
                correct=r
            )
            if r == 0:
                assert o == 0, type
            else:
                assert abs(o - r)/r < max_e, type

if __name__ == "__main__":
    tyre_model_test()
