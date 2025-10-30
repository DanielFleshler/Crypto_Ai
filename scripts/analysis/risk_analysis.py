#!/usr/bin/env python3
"""
Analyze impact of different risk per trade levels
"""

# Backtest results
win_rate = 0.3347  # 33.47%
avg_rr = 3.0  # 1:3 risk-reward
total_trades = 43
consecutive_losses_observed = 11  # From 2022 backtest

print("="*70)
print("RISK PER TRADE ANALYSIS")
print(f"Based on backtest: {total_trades} trades, {win_rate*100:.2f}% win rate")
print("="*70)
print()

initial_balance = 10000

risk_levels = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

for risk_pct in risk_levels:
    print(f"Risk Per Trade: {risk_pct*100}%")
    print("-" * 50)

    # Calculate drawdown from consecutive losses
    balance = initial_balance
    for i in range(consecutive_losses_observed):
        loss = balance * risk_pct
        balance -= loss

    drawdown_pct = ((initial_balance - balance) / initial_balance) * 100

    # Calculate potential profit from wins
    # With 33.47% win rate and 3:1 R:R
    expected_wins = total_trades * win_rate
    expected_losses = total_trades * (1 - win_rate)

    # Expected value per trade
    ev_per_trade = (win_rate * (risk_pct * avg_rr)) - ((1-win_rate) * risk_pct)
    total_ev = ev_per_trade * total_trades * initial_balance

    print(f"  Worst observed drawdown ({consecutive_losses_observed} losses): {drawdown_pct:.2f}%")
    print(f"  Balance after {consecutive_losses_observed} losses: ${balance:.2f}")

    if drawdown_pct > 50:
        print(f"  ⚠️  EXTREME RISK: >50% drawdown likely")
    elif drawdown_pct > 30:
        print(f"  ⚠️  HIGH RISK: >30% drawdown likely")
    elif drawdown_pct > 15:
        print(f"  ⚠️  MODERATE RISK: 15-30% drawdown possible")
    else:
        print(f"  ✓ LOW RISK: <15% drawdown")

    print(f"  Expected value (43 trades): ${total_ev:+.2f} ({(total_ev/initial_balance)*100:+.2f}%)")
    print()

print("="*70)
print("KELLY CRITERION OPTIMAL RISK")
print("="*70)

# Kelly formula: f = (bp - q) / b
# where: b = odds (3 for 3:1), p = win rate, q = loss rate
b = avg_rr
p = win_rate
q = 1 - p

kelly_fraction = (b * p - q) / b
half_kelly = kelly_fraction / 2

print(f"Win Rate: {p*100:.2f}%")
print(f"Risk-Reward: 1:{avg_rr}")
print(f"Full Kelly: {kelly_fraction*100:.2f}% per trade")
print(f"Half Kelly (recommended): {half_kelly*100:.2f}% per trade")
print()

if kelly_fraction <= 0:
    print("⚠️  WARNING: Negative Kelly = Strategy has negative expectation!")
    print("    Don't trade this strategy until win rate or R:R improves")
else:
    print(f"Recommendation: Risk between {max(0.01, half_kelly)*100:.1f}% and {kelly_fraction*100:.1f}% per trade")

print()
print("="*70)
print("CONCLUSION")
print("="*70)
print("Current system: 2% risk per trade")
print("Your request: 10% risk per trade")
print()

if risk_pct == 0.10:
    if kelly_fraction > 0 and 0.10 <= kelly_fraction * 2:
        print("✓ 10% is within acceptable range for this strategy")
    else:
        print("⚠️  10% is TOO AGGRESSIVE for current win rate")
        print(f"    Recommended max: {min(0.05, kelly_fraction)*100:.1f}%")

print()
print("With 33% win rate and current performance:")
print(f"  • 11 consecutive losses would cause {((1 - (1-0.10)**11)*100):.1f}% drawdown at 10% risk")
print(f"  • Probability of 11 consecutive losses: {((1-win_rate)**11)*100:.3f}%")
print(f"  • This WILL happen eventually with enough trades")
