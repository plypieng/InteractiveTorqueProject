# app/plots/plot_factory.py
import plotly.graph_objs as go

def create_normal_plot(x, y, y_axis_range, x2=None, y2=None, name1='File 1', name2='File 2'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name1, line=dict(width=1.5, color='#58a6ff')))
    if x2 is not None and y2 is not None:
        fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name=name2, line=dict(width=1.5, color='#f85149', dash='dot')))
    
    fig.update_layout(
        title='Torque Data Comparison',
        xaxis_title='X[mm]',
        yaxis_title='N[Ncm]',
        yaxis=dict(range=y_axis_range, gridcolor='rgba(48, 54, 61, 0.5)'),
        xaxis=dict(gridcolor='rgba(48, 54, 61, 0.5)'),
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c9d1d9'),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def create_filtered_plot(x, y_filtered, filtered_rms, moving_max_avg, moving_min_avg, cutoff_freq, y_axis_range, x2=None, y2_filtered=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_filtered, mode='lines', name='F1 Filtered', opacity=0.7))
    fig.add_trace(go.Scatter(x=x, y=filtered_rms, mode='lines', name='F1 RMS', line=dict(width=2)))
    
    if x2 is not None and y2_filtered is not None:
        fig.add_trace(go.Scatter(x=x2, y=y2_filtered, mode='lines', name='F2 Filtered', opacity=0.5, line=dict(dash='dot')))
    
    fig.add_trace(go.Scatter(x=x, y=moving_max_avg, mode='lines', name='F1 Max Avg', line=dict(color='#238636', width=1)))
    fig.add_trace(go.Scatter(x=x, y=moving_min_avg, mode='lines', name='F1 Min Avg', line=dict(color='#da3633', width=1)))
    
    fig.update_layout(
        title=f'High-Pass Filtered Comparison (Cutoff = {cutoff_freq} Hz)',
        xaxis_title='X[mm]',
        yaxis_title='N[Ncm]',
        hovermode='x unified',
        yaxis=dict(range=y_axis_range, gridcolor='rgba(48, 54, 61, 0.5)'),
        xaxis=dict(gridcolor='rgba(48, 54, 61, 0.5)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c9d1d9'),
    )
    return fig

def create_fft_plot(xf, amplitudes, xf2=None, amplitudes2=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=amplitudes, mode='lines', name='F1 FFT', fill='tozeroy'))
    if xf2 is not None and amplitudes2 is not None:
         fig.add_trace(go.Scatter(x=xf2, y=amplitudes2, mode='lines', name='F2 FFT', line=dict(dash='dot')))
    
    fig.update_layout(
        title='FFT Spectrum Comparison',
        xaxis_title='Frequency [Hz]',
        yaxis_title='Amplitude',
        hovermode='x unified',
        yaxis=dict(range=[0, min(0.1, max(amplitudes.max(), 0.05))], gridcolor='rgba(48, 54, 61, 0.5)'),
        xaxis=dict(gridcolor='rgba(48, 54, 61, 0.5)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c9d1d9'),
    )
    return fig