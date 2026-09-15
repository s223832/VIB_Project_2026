
def relative_diff(omega_num, omega_meas):

    # Eigenvalues
    lambda_num = omega_num**2
    lambda_meas = omega_meas**2

    lamda_rel = (lambda_num - lambda_meas) / lambda_meas

    return lamda_rel