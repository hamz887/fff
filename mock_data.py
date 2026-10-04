# -*- coding: utf-8 -*-
"""mock_data.py — synthetic Edge-IIoTset / CICIoT2023 files for the mock run (pipeline check only; the numbers have
no scientific meaning)."""
import numpy as np
import pandas as pd

EDGE_COLS = ["frame.time", "ip.src_host", "ip.dst_host", "arp.dst.proto_ipv4", "arp.opcode", "arp.hw.size",
             "arp.src.proto_ipv4", "icmp.checksum", "icmp.seq_le", "icmp.transmit_timestamp", "icmp.unused",
             "http.file_data", "http.content_length", "http.request.uri.query", "http.request.method", "http.referer",
             "http.request.full_uri", "http.request.version", "http.response", "http.tls_port", "tcp.ack", "tcp.ack_raw",
             "tcp.checksum", "tcp.connection.fin", "tcp.connection.rst", "tcp.connection.syn", "tcp.connection.synack",
             "tcp.dstport", "tcp.flags", "tcp.flags.ack", "tcp.len", "tcp.options", "tcp.payload", "tcp.seq",
             "tcp.srcport", "udp.port", "udp.stream", "udp.time_delta", "dns.qry.name", "dns.qry.name.len", "dns.qry.qu",
             "dns.qry.type", "dns.retransmission", "dns.retransmit_request", "dns.retransmit_request_in",
             "mqtt.conack.flags", "mqtt.conflag.cleansess", "mqtt.conflags", "mqtt.hdrflags", "mqtt.len",
             "mqtt.msg_decoded_as", "mqtt.msg", "mqtt.msgtype", "mqtt.proto_len", "mqtt.protoname", "mqtt.topic",
             "mqtt.topic_len", "mqtt.ver", "mbtcp.len", "mbtcp.trans_id", "mbtcp.unit_id", "Attack_label", "Attack_type"]
TYPES = ["DDoS_UDP", "DDoS_ICMP", "SQL_injection", "Password", "Vulnerability_scanner", "DDoS_TCP", "DDoS_HTTP",
         "Uploading", "Backdoor", "Port_Scanning", "XSS", "Ransomware", "MITM", "Fingerprinting"]


def mock_edge(n=9000, seed=0):
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < .3).astype(int)
    at = np.where(y == 1, rng.choice(TYPES, n), "Normal")
    df = pd.DataFrame(0.0, index=range(n), columns=EDGE_COLS)
    df = df.astype({c: object for c in ("frame.time", "ip.src_host", "http.request.method", "mqtt.protoname",
                                        "tcp.payload", "http.request.full_uri", "Attack_type")})
    df["frame.time"] = "2021-11-08"
    df["ip.src_host"] = np.where(y == 1, "192.168.0.170", "192.168.0.128")
    df["tcp.len"] = np.round(rng.gamma(2, 50, n) + y * rng.gamma(2, 30, n))
    df["tcp.flags"] = rng.choice([16, 24, 2, 18], n)
    df["tcp.flags.ack"] = (df["tcp.flags"] != 2).astype(float)
    df["tcp.connection.syn"] = np.isin(at, ["DDoS_TCP", "Port_Scanning"]) * 1.0
    df["tcp.connection.rst"] = np.isin(at, ["Port_Scanning", "Fingerprinting"]) * (rng.rand(n) < .6)
    df["icmp.checksum"] = np.where(at == "DDoS_ICMP", rng.randint(1, 65535, n), 0)
    df["udp.stream"] = np.where(at == "DDoS_UDP", rng.randint(1, 10 ** 6, n), rng.randint(0, 500, n))
    df["udp.time_delta"] = rng.exponential(.01, n)
    df["tcp.seq"] = np.round(rng.exponential(1e4, n))
    df["tcp.ack"] = np.round(rng.exponential(1e4, n))
    df["arp.opcode"] = np.where(at == "MITM", rng.choice([1, 2], n), 0)
    df["mqtt.len"] = np.where(y == 0, rng.choice([0, 12, 39], n), 0)
    for i, t in enumerate(TYPES):                                  # each type gets a weak signature
        df.loc[at == t, "mqtt.topic_len"] = rng.normal(i * 3, 2.5, (at == t).sum())
    df["http.content_length"] = np.where(np.isin(at, ["SQL_injection", "XSS", "Uploading"]), rng.randint(50, 800, n), 0)
    df["http.request.method"] = np.where(np.isin(at, ["SQL_injection", "XSS", "Uploading", "DDoS_HTTP"]),
                                         rng.choice(["GET", "POST"], n), rng.choice(["0", "GET"], n, p=[.9, .1]))
    df["mqtt.protoname"] = np.where((y == 0) & (rng.rand(n) < .3), "MQTT", np.where(y == 0, "0", "0.0"))
    # high-cardinality text with real missing values (exercises the structural encoder and the NaN handling)
    payload = np.array([f"{rng.randint(0, 16 ** 6):06x}{'ff' * rng.randint(0, 20)}" for _ in range(n)], dtype=object)
    payload[rng.rand(n) < .3] = np.nan
    df["tcp.payload"] = payload
    uri = np.array([f"/dvwa/?id={rng.randint(10 ** 6)}'--" for _ in range(n)], dtype=object)
    uri[~np.isin(at, ["SQL_injection", "XSS"])] = np.nan
    df["http.request.full_uri"] = uri
    df["Attack_label"], df["Attack_type"] = y, at
    return pd.concat([df, df.sample(frac=.05, random_state=seed)], ignore_index=True)


def mock_cic(n=6000, seed=1):
    rng = np.random.RandomState(seed)
    y = rng.rand(n) < .6
    lab = np.where(y, rng.choice(["DDoS-SYN_Flood", "DDoS-UDP_Flood", "DoS-TCP_Flood", "Recon-PortScan"], n),
                   "BenignTraffic")
    df = pd.DataFrame({c: rng.rand(n) * .2 for c in ["fin_flag_number", "syn_flag_number", "rst_flag_number",
                                                      "ack_flag_number", "HTTP", "DNS", "ICMP", "ARP"]})
    df["syn_flag_number"] = np.where(np.char.find(lab.astype(str), "SYN") >= 0, 1.0, df["syn_flag_number"])
    df["TCP"] = np.where(np.char.find(lab.astype(str), "UDP") >= 0, 0.0, 1.0)
    df["UDP"] = 1 - df["TCP"]
    df["Rate"] = rng.exponential(100, n)
    df["label"] = lab
    return df
