from candle_builder import CandleBuilder

builder = CandleBuilder()

payload = {

    "k": {

        "t": 100,
        "T": 200,

        "o": "10",
        "h": "20",
        "l": "5",
        "c": "15",

        "v": "100",

        "x": True

    }

}

print(

    builder.process_kline(
        "5m",
        payload
    )

)