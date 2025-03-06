import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

plt.style.use('dark_background')

# ----------------------------
# Data Generation and True Model
# ----------------------------
def generate_data(num_points=100, noise_std=0.8, random_seed=42):
    np.random.seed(random_seed)
    x_data = np.linspace(-1, 1, num_points)
    e = np.random.normal(0, noise_std, num_points)
    m_true = 0.7
    b_true = 0.2
    def true_model(x, m=m_true, b=b_true):
        return m * x + b + np.sin(5 * np.pi * x * m)
    y_true = true_model(x_data)
    y_data = y_true + e
    return x_data, y_data, true_model

# ----------------------------
# Log-Likelihood Function
# ----------------------------
def log_likelihood(x_data, y_data, m, b, sigma=0.1, model_func=None):
    if model_func is None:
        model_func = lambda x, m, b: m*x + b + np.sin(5*np.pi*x*m)
    log_p = np.log(1/(np.sqrt(2*np.pi)*sigma)) - 0.5*(y_data - model_func(x_data, m, b))**2/sigma**2
    return np.sum(log_p)

# ----------------------------
# Plotting Functions
# ----------------------------
def plot_model_fit(x_data, y_data, model_func, final_params):
    mhat, bhat = final_params
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x_data, y_data, label='Noisy Data', alpha=0.6)
    # Plot the noise-free true model using default parameters
    ax.plot(x_data, model_func(x_data), color='red', lw=2, label='True Model')
    # Plot the fitted model using current estimated parameters
    fitted = model_func(x_data, mhat, bhat)
    ax.plot(x_data, fitted, color='green', linestyle='--', lw=2, label='Fitted Model')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Model Fit: Data vs True and Fitted Model')
    ax.legend()
    return fig

def plot_trace(chain):
    chain = np.array(chain)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(chain[:, 0], label="m (slope)")
    ax.plot(chain[:, 1], label="b (intercept)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Parameter Value")
    ax.set_title("Trace Plot of Parameter Samples")
    ax.legend()
    return fig

# ----------------------------
# Streamlit App Interface
# ----------------------------
st.title("Metropolis Hastings Sampling Demo")
st.write("Model : $y(x) = m x + b + \sin{(5 \pi m x} ) + \epsilon $")

# Sidebar inputs for data and sampling parameters
st.sidebar.header("Data and Sampling Parameters")
num_points = st.sidebar.slider("Number of Data Points", min_value=50, max_value=500, value=100, step=10)
noise_std = st.sidebar.slider("Noise Std Dev", min_value=0.1, max_value=2.0, value=0.8, step=0.1)
iterations = st.sidebar.slider("Number of Iterations", min_value=50, max_value=500, value=100, step=10)
sigma = st.sidebar.slider("Sigma (Likelihood)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
init_m = st.sidebar.number_input("Initial m (slope)", value=1.0, step=0.1)
init_b = st.sidebar.number_input("Initial b (intercept)", value=1.0, step=0.1)
run_sampling = st.sidebar.button("Run Sampling")

if run_sampling:
    # Generate data using the custom true model
    x_data, y_data, true_model_func = generate_data(num_points=num_points, noise_std=noise_std)
    
    # Create placeholders for live plot and progress text
    plot_placeholder = st.empty()
    progress_text = st.empty()
    
    # Initialize parameters and chain for sampling
    mhat = init_m
    bhat = init_b
    log_p = log_likelihood(x_data, y_data, mhat, bhat, sigma, true_model_func)
    chain = [(mhat, bhat)]
    T = 1000.0  # initial temperature

    # Run the sampling loop with live updates
    for i in range(iterations):
        T *= 0.99  # cooling schedule
        mnew = np.random.normal(mhat, 0.5)
        bnew = np.random.normal(bhat, 0.5)
        log_prob_new = log_likelihood(x_data, y_data, mnew, bnew, sigma, true_model_func)
        
        if log_prob_new > log_p:
            mhat, bhat = mnew, bnew
            log_p = log_prob_new
        else:
            u = np.random.rand()
            if u < np.exp((log_prob_new - log_p) / T):
                mhat, bhat = mnew, bnew
                log_p = log_prob_new
        
        chain.append((mhat, bhat))
        
        # Update the live plot every 10 iterations
        if i % 10 == 0:
            fig = plot_model_fit(x_data, y_data, true_model_func, (mhat, bhat))
            plot_placeholder.pyplot(fig)
            progress_text.write(f"Iteration: {i}, Temperature: {T:.2f}")
            time.sleep(0.1)  # small delay for visualization
    
    # After the loop finishes, display final model fit plot
    st.write("### Final Parameter Estimates")
    st.write(f"Estimated m : {mhat:.3f}, True m : 0.7 ")
    st.write(f"Estimated b : {bhat:.3f}, True b : 0.2 ")
    # final_fig = plot_model_fit(x_data, y_data, true_model_func, (mhat, bhat))
    # st.pyplot(final_fig)
    
    # Display the trace plot of the parameter samples
    st.subheader("Parameter Trace Plot")
    trace_fig = plot_trace(chain)
    st.pyplot(trace_fig)