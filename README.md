# Volatility Modeling with Hidden Markov Model GARCH (HMM-GARCH)

A comprehensive implementation of Hidden Markov Model combined with GARCH for financial volatility modeling. This framework identifies unobservable market regimes and models regime-specific volatility dynamics using state-of-the-art time series techniques.

## Overview

Financial markets exhibit volatility clustering and regime-switching behavior - periods of high volatility followed by periods of low volatility. Traditional GARCH models assume constant parameters, but HMM-GARCH allows parameters to switch between different market states (regimes) according to an unobservable Markov chain.

### Key Features

- **Hidden Markov Model**: Identifies latent market regimes using Forward-Backward algorithm
- **Regime-Specific GARCH**: Separate GARCH(1,1) parameters for each regime
- **EM Algorithm**: Maximum likelihood estimation via Expectation-Maximization
- **Viterbi Decoding**: Most likely sequence of hidden states
- **Transition Dynamics**: Estimates regime persistence and switching probabilities
- **Data Pipeline**: Automated download, cleaning, and preprocessing of financial data
- **Model Comparison**: Performance evaluation against standard GARCH and rolling volatility

## Methodology

### 1. Hidden Markov Model Framework

The HMM-GARCH model assumes:
- **Hidden States**: Market regimes (e.g., low volatility, high volatility)
- **Observations**: Daily return series
- **State-Dependent Emissions**: Each regime has different GARCH parameters
- **Markov Property**: Current regime depends only on previous regime

### 2. Model Specification

**State Equation:**
```
S_t | S_{t-1} ~ Categorical(π_{S_{t-1}})
```

**Observation Equation:**
```
r_t | S_t = k ~ N(μ_k, σ²_{k,t})
σ²_{k,t} = ω_k + α_k * r²_{t-1} + β_k * σ²_{k,t-1}
```

Where:
- `S_t` is the hidden regime at time t
- `π` is the transition probability matrix
- `ω_k, α_k, β_k` are regime-specific GARCH parameters

### 3. Estimation Process

1. **Initialization**: Set initial parameters for transition matrix and GARCH coefficients
2. **E-Step**: Forward-backward algorithm to compute regime probabilities
3. **M-Step**: Update parameters via weighted maximum likelihood
4. **Convergence**: Iterate until log-likelihood converges
5. **Decoding**: Viterbi algorithm for most likely state sequence

## Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn arch yfinance
```