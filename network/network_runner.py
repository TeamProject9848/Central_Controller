import asyncio
import threading

from network.websocket_server import CompanionWebSocketServer


def start_network(controller):

    async def run():

        ws_server = CompanionWebSocketServer(
            controller
        )

        controller.set_flutter_server(
            ws_server.bridge
        )

        await ws_server.start()

        await asyncio.Future()

    def thread_target():
        asyncio.run(run())

    thread = threading.Thread(
        target=thread_target,
        daemon=True,
        name="FlutterNetwork"
    )

    thread.start()

    return thread