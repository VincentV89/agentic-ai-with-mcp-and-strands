#!/usr/bin/env python

import argparse
import json

from deploy_to_agentcore import deploy_agentcore_with_cognito_jwt
from utils01 import setup_cognito_user_pool, reauthenticate_user


discovery_url = None
client_id = None


def step_01_setup_cognito():
    print("Setting up Amazon Cognito user pool...")
    cognito_config = setup_cognito_user_pool()
    print("Cognito setup completed ✓")
    discovery_url = cognito_config.get("discovery_url")
    client_id = cognito_config.get("client_id")
    return client_id, discovery_url


def step_02_deployagentcore(client_id, discovery_url):
    return deploy_agentcore_with_cognito_jwt(
        agent_name="strands_agent_inbound_identity",
        entry_point="strands_claude.py",
        discovery_url=discovery_url,
        client_id=client_id
    )


def print_header(header):
    print()
    print('=' * 80 + '\n' + header + '\n' + '=' * 80)


def main():
    client_id, discovery_url = step_01_setup_cognito()
    launch_result, agentcore_runtime = step_02_deployagentcore(client_id, discovery_url)

    print_header('Invoking AgentCore Runtime without authorization')
    try:
        invoke_response = agentcore_runtime.invoke({"prompt": "How is the weather now?"})
        print(json.dumps(invoke_response, indent=2, default=str))
    except Exception as e:
        print(f"Error invoking AgentCore: {e}")
        print()

    print_header('Invoking AgentCore Runtime with authorization')
    bearer_token = reauthenticate_user(client_id)

    try:
        invoke_response = agentcore_runtime.invoke(
            {"prompt": "How is the weather now?"}, 
            bearer_token=bearer_token
        )
        print(json.dumps(invoke_response, indent=2, default=str))
    except Exception as e:
        print(f"Error invoking AgentCore: {e}")
        print()


if __name__ == '__main__':
    main()
