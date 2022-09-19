"""Command Line Processor for robot WiFi configuration"""
from asyncio.log import logger
import requests
import argparse
import logging


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        #usage="%(prog)s [OPTION] [FILE]...",
        description="Configure wifi."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 0.2.0"
    )
    parser.add_argument(
        "-s", "--ssid",
        dest='ssid',
        help='the SSID (network identification)'
    )
    parser.add_argument(
        "-a", "--address", default="192.168.4.1",
        dest='address',
        help='the host api address (default=192.168.4.1)'
    )
    parser.add_argument(
        "-p", "--password",
        dest='psw',
        help='the network pass phrase'
    )
    parser.add_argument(
        "action",
        choices=["list", "show", "act", "inact"],
        help='the action'
    )
    return parser


def main():
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger("wheelly.envs.robot").setLevel(logging.DEBUG)

    args = init_argparse().parse_args()

    if args.action == "list":
        api_url = f"http://{args.address}/api/v1/wheelly/networks"
        response = requests.get(api_url)
        json = response.json()
        if response.status_code != 200:
            logging.error(f"Error {response.status_code}, {json}")
        else:
            print("Networks:")
            for ssid in json["networks"]:
                if ssid is not None:
                    print(f'  "{ssid}"')
    if args.action == "show":
        api_url = f"http://{args.address}/api/v1/wheelly/networks/network"
        response = requests.get(api_url)
        json = response.json()
        if response.status_code != 200:
            logging.error(f"Error {response.status_code}, {json}")
        else:
            print("Configuration:")
            print(f' Status: {"active" if json["active"] else "inactive"}')
            print(f' Network SSID: "{json["ssid"]}"')
            print(" Password: ***")
    elif args.action == "act":
        api_url = f'http://{args.address}/api/v1/wheelly/networks/network'
        if args.ssid is None:
            logging.error("Missing SSID")
            raise Exception("Missing SSID")
        if args.psw is None:
            raise Exception("Missing pass phrase")
        body = {
            "active": True,
            "ssid": args.ssid,
            "password": args.psw
        }
        response = requests.post(api_url, json=body)
        json = response.json()
        if response.status_code != 200:
            logging.error(f"Error {response.status_code}, {json}")
        else:
            print(f"Activated network: {args.ssid}")
            print(f"Restart wheely required to reload new configuration.")
    elif args.action == "inact":
        api_url = f"http://{args.address}/api/v1/wheelly/networks/network"
        if args.ssid is None:
            logging.error("Missing SSID")
            raise Exception("Missing SSID")
        if args.psw is None:
            raise Exception("Missing pass phrase")
        body = {
            "active": False,
            "ssid": args.ssid,
            "password": args.psw
        }
        response = requests.post(api_url, json=body)
        json = response.json()
        if response.status_code != 200:
            logging.error(f"Error {response.status_code}, {json}")
        else:
            print(f"Inactivated network: {args.ssid}")
            print("Restart wheely required to reload new configuration.")
            print(f'Wheelly will act as access point for the "Wheelly" network without pass phrase.')


if __name__ == '__main__':
    main()
