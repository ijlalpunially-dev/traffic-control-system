def traffic_decision(vehicle_count, emergency_detected):
    if emergency_detected:
        return "🚑 Emergency vehicle detected → GREEN for emergency lane!"
    if vehicle_count > 15:
        return "Heavy traffic → Extend GREEN light duration."
    if 5 < vehicle_count <= 15:
        return "Moderate traffic → Normal GREEN cycle."
    return "Light traffic → Switch lights normally."
