# plots.py

import plotly.graph_objs as go

def create_normal_plot(x, y, y_axis_range):
    """
    Create a Plotly figure for the normal data plot.

    Parameters:
    - x (array-like): X-axis data.
    - y (array-like): Y-axis data.
    - y_axis_range (list): Y-axis range.

    Returns:
    - fig (go.Figure): Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
    fig.update_layout(
        title='Normal Data Plot',
        xaxis_title='X[mm]',
        yaxis_title='N[Ncm]',
        yaxis=dict(range=y_axis_range),
    )
    return fig

def create_filtered_plot(x, y_filtered, filtered_rms, cutoff_freq):
    """
    Create a Plotly figure for the high-pass filtered data plot.

    Parameters:
    - x (array-like): X-axis data.
    - y_filtered (array-like): Filtered Y-axis data.
    - filtered_rms (array-like): RMS values.
    - cutoff_freq (float): Cutoff frequency used.

    Returns:
    - fig (go.Figure): Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_filtered, mode='lines', name='Filtered Data'))
    fig.add_trace(go.Scatter(x=x, y=filtered_rms, mode='lines', name='RMS'))
    fig.update_layout(
        title=f'High-Pass Filtered Data (Cutoff = {cutoff_freq} Hz)',
        xaxis_title='X[mm]',
        yaxis_title='N[Ncm]',
        yaxis=dict(range=[-1, 1]),
    )
    return fig

def create_fft_plot(xf, amplitudes):
    """
    Create a Plotly figure for the FFT plot.

    Parameters:
    - xf (array-like): Frequencies.
    - amplitudes (array-like): Amplitudes.

    Returns:
    - fig (go.Figure): Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=amplitudes, mode='lines'))
    fig.update_layout(
        title='FFT of Data',
        xaxis_title='Frequency [Hz]',
        yaxis_title='Amplitude',
        yaxis=dict(range=[0, 0.05]),
    )
    return fig
