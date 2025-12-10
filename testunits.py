c = 2.99792458
h = 6.62607015
me = 9.1093837
mp = 1.67262192

c_exp = 8
h_exp = -24
me_exp = -31
mp_exp = -37

alpha = me**4 * c**5 / h**3
alpha_exp = me_exp * 4 + c_exp * 5 - h_exp * 3

print(f"{alpha} *10^{alpha_exp} ")

beta = h / (mp ** (1.0 / 3.0) * me * c)
_3beta_exp = 3 * h_exp - mp_exp - 3 * me_exp - 3 * c_exp

print(f"{beta} *10^{_3beta_exp}/3 ")
