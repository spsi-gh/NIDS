# Dataset inspection

- Training records: 175,341; test records: 82,332.
- Columns: 45; usable model features: 42.
- Training normal: 56,000; attacks: 119,341.
- Test normal: 37,000; attacks: 45,332.
- Missing values in original training CSV: 0; test CSV: 0.

## Column types

| Column | pandas dtype | Training missing |
|---|---|---:|
| id | int64 | 0 |
| dur | float64 | 0 |
| proto | str | 0 |
| service | str | 0 |
| state | str | 0 |
| spkts | int64 | 0 |
| dpkts | int64 | 0 |
| sbytes | int64 | 0 |
| dbytes | int64 | 0 |
| rate | float64 | 0 |
| sttl | int64 | 0 |
| dttl | int64 | 0 |
| sload | float64 | 0 |
| dload | float64 | 0 |
| sloss | int64 | 0 |
| dloss | int64 | 0 |
| sinpkt | float64 | 0 |
| dinpkt | float64 | 0 |
| sjit | float64 | 0 |
| djit | float64 | 0 |
| swin | int64 | 0 |
| stcpb | int64 | 0 |
| dtcpb | int64 | 0 |
| dwin | int64 | 0 |
| tcprtt | float64 | 0 |
| synack | float64 | 0 |
| ackdat | float64 | 0 |
| smean | int64 | 0 |
| dmean | int64 | 0 |
| trans_depth | int64 | 0 |
| response_body_len | int64 | 0 |
| ct_srv_src | int64 | 0 |
| ct_state_ttl | int64 | 0 |
| ct_dst_ltm | int64 | 0 |
| ct_src_dport_ltm | int64 | 0 |
| ct_dst_sport_ltm | int64 | 0 |
| ct_dst_src_ltm | int64 | 0 |
| is_ftp_login | int64 | 0 |
| ct_ftp_cmd | int64 | 0 |
| ct_flw_http_mthd | int64 | 0 |
| ct_src_ltm | int64 | 0 |
| ct_srv_dst | int64 | 0 |
| is_sm_ips_ports | int64 | 0 |
| attack_cat | str | 0 |
| label | int64 | 0 |

## Training attack categories

| Category | Rows |
|---|---:|
| Normal | 56,000 |
| Generic | 40,000 |
| Exploits | 33,393 |
| Fuzzers | 18,184 |
| DoS | 12,264 |
| Reconnaissance | 10,491 |
| Analysis | 2,000 |
| Backdoor | 1,746 |
| Shellcode | 1,133 |
| Worms | 130 |

Charts: [Open interactive EDA charts](eda_charts.html).
