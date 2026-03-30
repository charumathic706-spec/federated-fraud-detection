from __future__ import annotations

import json
from pathlib import Path

from web3 import Web3


RPC_URL = "http://127.0.0.1:8545"
THIS_DIR = Path(__file__).resolve().parent
DEPLOY_FILE = THIS_DIR / "eth_deployment.json"


def main() -> None:
    with DEPLOY_FILE.open("r", encoding="utf-8") as f:
        deployment = json.load(f)

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    print(f"RPC URL:   {RPC_URL}")
    print(f"Connected: {w3.is_connected()}")

    if not w3.is_connected():
        print("Ganache is not reachable. Start Ganache and try again.")
        return

    print(f"Chain ID:  {w3.eth.chain_id}")

    address = Web3.to_checksum_address(deployment["address"])
    abi = deployment["abi"]
    code = w3.eth.get_code(address)

    print(f"Address:   {address}")
    print(f"Has code:  {code != b''}")

    if code == b"":
        print("No contract code found at this address on the current chain.")
        print("Ganache may have been restarted, so you may need to redeploy.")
        return

    contract = w3.eth.contract(address=address, abi=abi)

    try:
        print(f"Owner:       {contract.functions.owner().call()}")
        print(f"Round count: {contract.functions.getRoundCount().call()}")
        print(f"Alert count: {contract.functions.getAlertCount().call()}")
        print(f"Audit count: {contract.functions.getAuditCount().call()}")
        intact, broken_at = contract.functions.verifyFullChain().call()
        print(f"Full chain:  intact={intact}, brokenAtRound={broken_at}")
    except Exception as exc:
        print(f"Contract call failed: {exc}")
        print("The deployment file may not match the current Ganache chain state.")


if __name__ == "__main__":
    main()
