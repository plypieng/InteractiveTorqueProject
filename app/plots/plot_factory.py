# app/plots/plot_factory.py
import plotly.graph_objs as go

def create_normal_plot(x, y, y_axis_range):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Normal Data'))
    fig.update_layout(
        title='Normal Data Plot',
        xaxis_title='X[mm]',
        yaxis_title='N[Ncm]',
        yaxis=dict(range=y_axis_range),
        hovermode='closest',
    )
    return fig

def create_filtered_plot(x, y_filtered, filtered_rms, moving_max_avg, moving_min_avg, cutoff_freq, y_axis_range):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_filtered, mode='lines', name='Filtered Data'))
    fig.add_trace(go.Scatter(x=x, y=filtered_rms, mode='lines', name='RMS'))
    fig.add_trace(go.Scatter(x=x, y=moving_max_avg, mode='lines', name='Moving Avg of Max', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=x, y=moving_min_avg, mode='lines', name='Moving Avg of Min', line=dict(color='red')))
    fig.update_layout(
        title=f'High-Pass Filtered Data (Cutoff = {cutoff_freq} Hz)',
        xaxis_title='X[mm]',
        yaxis_title='N[Ncm]',
        hovermode='closest',
        yaxis=dict(range=y_axis_range),
    )
    return fig

def create_fft_plot(xf, amplitudes):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=amplitudes, mode='lines', name='FFT'))
    fig.update_layout(
        title='FFT of Data',
        xaxis_title='Frequency [Hz]',
        yaxis_title='Amplitude',
        hovermode='closest',
        yaxis=dict(range=[0, 0.05]),
    )
    return fig
