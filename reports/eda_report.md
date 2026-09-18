# Phase 3: Exploratory Data Analysis (EDA) Report

## Flight Delay Analysis Summary
- **Total Flights Analyzed**: 1200 records.
- **Overall Mean Delay**: 51.87 minutes (Std: 23.43 mins).
- **Delay Risk Breakdown**:
  - **On-Time / Low Risk (<15 min)**: 71 flights (5.9%)
  - **Moderate Delay Risk (15-45 min)**: 405 flights (33.8%)
  - **Severe Delay Risk (>45 min)**: 724 flights (60.3%)

### Key Weather & Operational Drivers
1. **Severe Weather Impact**: Thunderstorms and High Winds induce the highest average delay (72.5 min average delay).
2. **Airport Congestion**: Traffic index > 7.5 strongly correlates with exponential delay escalation (correlation coefficient: 0.38).
3. **Cascading Prior Leg Delays**: Inbound prior flight delays above 30 minutes cause downstream delays in over 85% of cases.

---

## Aircraft Inspection & Defect Frequency Analysis
- **Total Inspection Images**: 25 high-resolution component panels.
- **Total Detected Defect Instances**: 49.
- **Defect Frequencies**:
  - **Turbine Blade Erosion**: 11 instances
  - **Rivet Corrosion**: 10 instances
  - **Wing Surface Dent**: 9 instances
  - **Composite Delamination**: 11 instances
  - **Fuselage Crack**: 8 instances

- **Defect Severities**:
  - **Moderate**: 18 instances
  - **Critical**: 15 instances
  - **Minor**: 16 instances
