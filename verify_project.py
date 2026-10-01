from backend.ml import predict

tests = [
    ("The ticket machine charged my card twice for one journey", "Billing"),
    ("The driver was using a phone and driving dangerously", "Safety"),
    ("The bus arrived forty minutes late and I missed my class", "Operations"),
    ("The bus stop shelter is broken and needs repair", "Infrastructure"),
]
for text, expected in tests:
    out = predict(text)
    print(f"{expected:18} -> {out['prediction']:18} | {out['confidence']:.1%}")
    assert out["prediction"] == expected
print("All smoke tests passed.")
