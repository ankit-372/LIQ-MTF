from order_flow import OrderFlow

flow = OrderFlow()

flow.process_trade({
    "q": "2.5",
    "p": "58850",
    "m": False
})

flow.process_trade({
    "q": "1.0",
    "p": "58849",
    "m": True
})

print(flow.publish_snapshot())
