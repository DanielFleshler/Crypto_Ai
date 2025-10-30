#!/usr/bin/env python3
"""
Verify Risk-Reward and Position Sizing Calculations
"""

# Test scenario
balance = 10000
entry_price = 100
stop_loss = 98  # 2% below entry
take_profit = 106  # 6% above entry

# Current R:R calculation
risk = abs(entry_price - stop_loss)  # 2
reward = abs(take_profit - entry_price)  # 6
rr = reward / risk  # 6/2 = 3.0

print("="*70)
print("RISK-REWARD VERIFICATION")
print("="*70)
print(f"Balance: ${balance}")
print(f"Entry Price: ${entry_price}")
print(f"Stop Loss: ${stop_loss}")
print(f"Take Profit: ${take_profit}")
print()
print(f"Risk (entry - SL): ${risk}")
print(f"Reward (TP - entry): ${reward}")
print(f"R:R Ratio: {rr}")
print()
print(f"✓ R:R = {rr} means: Risk $1 to make ${rr}")
print(f"✓ This MEETS the 1:3 minimum requirement" if rr >= 3.0 else f"✗ This FAILS the 1:3 requirement")
print()

# Position sizing with 2% risk
risk_per_trade = 0.02  # Current config
risk_percentage = risk / entry_price  # 2/100 = 0.02 (2% distance to stop)

position_size_dollars = (balance * risk_per_trade) / risk_percentage
position_qty = position_size_dollars / entry_price

actual_risk_amount = position_qty * risk  # How much $ you lose if stopped out

print("="*70)
print("POSITION SIZING (Current: 2% risk)")
print("="*70)
print(f"Risk per trade: {risk_per_trade*100}%")
print(f"Risk percentage (distance to SL): {risk_percentage*100}%")
print(f"Position size: ${position_size_dollars:.2f}")
print(f"Position quantity: {position_qty:.4f}")
print(f"Actual risk amount if stopped out: ${actual_risk_amount:.2f}")
print(f"Risk as % of balance: {(actual_risk_amount/balance)*100:.2f}%")
print()

if abs(actual_risk_amount - (balance * risk_per_trade)) < 0.01:
    print("✓ Position sizing is CORRECT - risks exactly 2% of balance")
else:
    print(f"✗ Position sizing is WRONG - should risk ${balance * risk_per_trade} but risks ${actual_risk_amount}")

print()

# What if we want 10% risk?
risk_per_trade_target = 0.10
position_size_dollars_target = (balance * risk_per_trade_target) / risk_percentage
position_qty_target = position_size_dollars_target / entry_price
actual_risk_amount_target = position_qty_target * risk

print("="*70)
print("POSITION SIZING (Target: 10% risk)")
print("="*70)
print(f"Target risk per trade: {risk_per_trade_target*100}%")
print(f"Position size needed: ${position_size_dollars_target:.2f}")
print(f"Position quantity needed: {position_qty_target:.4f}")
print(f"Actual risk amount if stopped out: ${actual_risk_amount_target:.2f}")
print(f"Risk as % of balance: {(actual_risk_amount_target/balance)*100:.2f}%")
print()

if abs(actual_risk_amount_target - (balance * risk_per_trade_target)) < 0.01:
    print("✓ Would risk exactly 10% of balance")
else:
    print(f"✗ Would risk ${actual_risk_amount_target} instead of ${balance * risk_per_trade_target}")

print()

# Verify reward
potential_profit = position_qty_target * reward
print("="*70)
print("PROFIT/LOSS VERIFICATION (with 10% risk)")
print("="*70)
print(f"If stopped out: LOSE ${actual_risk_amount_target:.2f} ({(actual_risk_amount_target/balance)*100:.2f}%)")
print(f"If TP hit: WIN ${potential_profit:.2f} ({(potential_profit/balance)*100:.2f}%)")
print(f"Actual R:R achieved: {potential_profit / actual_risk_amount_target:.2f}")
print()

if abs((potential_profit / actual_risk_amount_target) - rr) < 0.01:
    print("✓ Actual R:R matches calculated R:R")
else:
    print(f"✗ Actual R:R {potential_profit / actual_risk_amount_target:.2f} doesn't match calculated {rr}")

print()
print("="*70)
print("SUMMARY")
print("="*70)
print(f"Current config: {risk_per_trade*100}% risk per trade")
print(f"User wants: {risk_per_trade_target*100}% risk per trade")
print(f"Change needed: config/trading_config.yaml")
print(f"  max_risk_per_trade: {risk_per_trade} → {risk_per_trade_target}")
