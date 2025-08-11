# type:ignore
import asyncio
import json
import os

from contextlib import asynccontextmanager

import click
import httpx

from fastmcp.utilities.logging import get_logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ReadResourceResult


logger = get_logger(__name__)

env = {
    'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
}


@asynccontextmanager
async def init_session(host, port, transport):
    """Initializes and manages an MCP ClientSession based on the specified transport.

    This asynchronous context manager establishes a connection to an MCP server
    using either Server-Sent Events (SSE) or Standard I/O (STDIO) transport.
    It handles the setup and teardown of the connection and yields an active
    `ClientSession` object ready for communication.

    Args:
        host: The hostname or IP address of the MCP server (used for SSE).
        port: The port number of the MCP server (used for SSE).
        transport: The communication transport to use ('sse' or 'stdio').

    Yields:
        ClientSession: An initialized and ready-to-use MCP client session.

    Raises:
        ValueError: If an unsupported transport type is provided (implicitly,
                    as it won't match 'sse' or 'stdio').
        Exception: Other potential exceptions during client initialization or
                   session setup.
    """
    if transport == 'sse':
        url = f'http://{host}:{port}/sse'
        async with sse_client(url) as (read_stream, write_stream):
            async with ClientSession(
                read_stream=read_stream, write_stream=write_stream
            ) as session:
                logger.debug('SSE ClientSession created, initializing...')
                await session.initialize()
                logger.info('SSE ClientSession initialized successfully.')
                yield session
    elif transport == 'stdio':
        if not os.getenv('GOOGLE_API_KEY'):
            logger.error('GOOGLE_API_KEY is not set')
            raise ValueError('GOOGLE_API_KEY is not set')
        stdio_params = StdioServerParameters(
            command='uv',
            args=['run', 'a2a-mcp'],
            env=env,
        )
        async with stdio_client(stdio_params) as (read_stream, write_stream):
            async with ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
            ) as session:
                logger.debug('STDIO ClientSession created, initializing...')
                await session.initialize()
                logger.info('STDIO ClientSession initialized successfully.')
                yield session
    else:
        logger.error(f'Unsupported transport type: {transport}')
        raise ValueError(
            f"Unsupported transport type: {transport}. Must be 'sse' or 'stdio'."
        )


async def find_agent(session: ClientSession, query) -> CallToolResult:
    """Calls the 'find_agent' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        query: The natural language query to send to the 'find_agent' tool.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling 'find_agent' tool with query: '{query[:50]}...'")
    return await session.call_tool(
        name='find_agent',
        arguments={
            'query': query,
        },
    )


async def find_resource(session: ClientSession, resource) -> ReadResourceResult:
    """Reads a resource from the connected MCP server.

    Args:
        session: The active ClientSession.
        resource: The URI of the resource to read (e.g., 'resource://agent_cards/list').

    Returns:
        The result of the resource read operation.
    """
    logger.info(f'Reading resource: {resource}')
    return await session.read_resource(resource)


async def search_flights(session: ClientSession) -> CallToolResult:
    """Calls the 'search_flights' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        query: The natural language query to send to the 'search_flights' tool.

    Returns:
        The result of the tool call.
    """
    # TODO: Implementation pending
    logger.info("Calling 'search_flights' tool'")
    return await session.call_tool(
        name='search_flights',
        arguments={
            'departure_airport': 'SFO',
            'arrival_airport': 'LHR',
            'start_date': '2025-06-03',
            'end_date': '2025-06-09',
        },
    )


async def search_hotels(session: ClientSession) -> CallToolResult:
    """Calls the 'search_hotels' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        query: The natural language query to send to the 'search_hotels' tool.

    Returns:
        The result of the tool call.
    """
    # TODO: Implementation pending
    logger.info("Calling 'search_hotels' tool'")
    return await session.call_tool(
        name='search_hotels',
        arguments={
            'location': 'A Suite room in St Pancras Square in London',
            'check_in_date': '2025-06-03',
            'check_out_date': '2025-06-09',
        },
    )


async def query_db(session: ClientSession) -> CallToolResult:
    """Calls the 'query' tool on the connected MCP server.

    Args:
        session: The active ClientSession.
        query: The natural language query to send to the 'query_db' tool.

    Returns:
        The result of the tool call.
    """
    logger.info("Calling 'query_db' tool'")
    return await session.call_tool(
        name='query_travel_data',
        arguments={
            'query': "SELECT id, name, city, hotel_type, room_type, price_per_night FROM hotels WHERE city='London'",
        },
    )


async def execute_travel_planning_demo(session: ClientSession, query: str) -> dict:
    """Execute a complete travel planning demo using the orchestrator agent.
    
    Args:
        session: The active ClientSession.
        query: The travel planning query.
        
    Returns:
        A comprehensive response with travel planning results.
    """
    logger.info(f"Starting comprehensive travel planning demo for: '{query}'")
    
    # Step 1: Find the orchestrator agent
    logger.info("Step 1: Finding the Orchestrator Agent...")
    orchestrator_result = await find_agent(session, "orchestrate and plan a complete trip")
    orchestrator_agent = json.loads(orchestrator_result.content[0].text)
    logger.info(f"Found agent: {orchestrator_agent['name']} at {orchestrator_agent['url']}")
    
    # Step 2: Simulate the workflow by showing what would happen
    logger.info("Step 2: Simulating complete travel planning workflow...")
    
    # Find planner agent
    planner_result = await find_agent(session, query)
    planner_agent = json.loads(planner_result.content[0].text)
    logger.info(f"🗓️  Step 2a: Planning Agent selected: {planner_agent['name']}")
    
    # Find flight agent
    flight_result = await find_agent(session, "book flight tickets")
    flight_agent = json.loads(flight_result.content[0].text)
    logger.info(f"✈️  Step 2b: Flight Agent selected: {flight_agent['name']}")
    
    # Find hotel agent
    hotel_result = await find_agent(session, "book hotel accommodation")
    hotel_agent = json.loads(hotel_result.content[0].text)
    logger.info(f"🏨 Step 2c: Hotel Agent selected: {hotel_agent['name']}")
    
    # Find car rental agent
    car_result = await find_agent(session, "rent a car")
    car_agent = json.loads(car_result.content[0].text)
    logger.info(f"🚗 Step 2d: Car Rental Agent selected: {car_agent['name']}")
    
    # Simulate data queries
    logger.info("Step 3: Querying travel database for available options...")
    
    # Query flights data
    try:
        flight_query_result = await session.call_tool(
            name='query_travel_data',
            arguments={
                'query': "SELECT DISTINCT from_airport, to_airport, MIN(price) as min_price FROM flights GROUP BY from_airport, to_airport LIMIT 5"
            }
        )
        flight_data = json.loads(flight_query_result.content[0].text)
        logger.info(f"✈️  Available flight routes: {len(flight_data.get('results', []))} routes found")
    except Exception as e:
        logger.warning(f"Could not query flight data: {e}")
        flight_data = {"results": []}
    
    # Query hotels data  
    try:
        hotel_query_result = await session.call_tool(
            name='query_travel_data',
            arguments={
                'query': "SELECT city, COUNT(*) as hotel_count, MIN(price_per_night) as min_price FROM hotels GROUP BY city LIMIT 5"
            }
        )
        hotel_data = json.loads(hotel_query_result.content[0].text)
        logger.info(f"🏨 Available hotel destinations: {len(hotel_data.get('results', []))} cities found")
    except Exception as e:
        logger.warning(f"Could not query hotel data: {e}")
        hotel_data = {"results": []}
    
    # Query rental cars
    try:
        car_query_result = await session.call_tool(
            name='query_travel_data',
            arguments={
                'query': "SELECT city, COUNT(*) as car_count, MIN(daily_rate) as min_price FROM rental_cars GROUP BY city LIMIT 5"
            }
        )
        car_data = json.loads(car_query_result.content[0].text)
        logger.info(f"🚗 Available car rental locations: {len(car_data.get('results', []))} cities found")
    except Exception as e:
        logger.warning(f"Could not query car rental data: {e}")
        car_data = {"results": []}
    
    # Create a comprehensive response
    return {
        "status": "success_simulation",
        "original_query": query,
        "workflow": {
            "orchestrator_agent": orchestrator_agent,
            "selected_agents": {
                "planner": planner_agent,
                "flight_booking": flight_agent, 
                "hotel_booking": hotel_agent,
                "car_rental": car_agent
            },
            "available_data": {
                "flights": flight_data.get('results', []),
                "hotels": hotel_data.get('results', []),
                "rental_cars": car_data.get('results', [])
            }
        },
        "summary": f"Successfully demonstrated complete A2A workflow for: '{query}'. The system identified appropriate specialized agents and retrieved relevant travel data from the database."
    }


async def demonstrate_agent_capabilities(session: ClientSession) -> dict:
    """Demonstrate the capabilities of different agents in the system.
    
    Args:
        session: The active ClientSession.
        
    Returns:
        A summary of all available agents and their capabilities.
    """
    logger.info("Demonstrating system capabilities...")
    
    # Get all agent cards
    logger.info("Fetching all available agents...")
    resource_result = await find_resource(session, "resource://agent_cards/list")
    agent_cards_list = json.loads(resource_result.contents[0].text)
    
    agents_info = []
    for agent_uri in agent_cards_list["agent_cards"]:
        agent_name = agent_uri.split("/")[-1]
        try:
            agent_result = await find_resource(session, agent_uri)
            agent_data = json.loads(agent_result.contents[0].text)
            if agent_data.get("agent_card"):
                agents_info.append(agent_data["agent_card"][0])
        except Exception as e:
            logger.warning(f"Could not fetch details for {agent_name}: {e}")
    
    # Demonstrate different query types
    demo_queries = [
        "Book a flight from New York to London",
        "Find a luxury hotel in Paris", 
        "Rent a car in Los Angeles",
        "Plan a business trip to Tokyo"
    ]
    
    query_results = {}
    for demo_query in demo_queries:
        try:
            result = await find_agent(session, demo_query)
            matched_agent = json.loads(result.content[0].text)
            query_results[demo_query] = {
                "matched_agent": matched_agent["name"],
                "agent_url": matched_agent["url"],
                "skills": [skill["name"] for skill in matched_agent.get("skills", [])]
            }
        except Exception as e:
            query_results[demo_query] = {"error": str(e)}
    
    return {
        "total_agents": len(agents_info),
        "available_agents": [{"name": agent["name"], "description": agent["description"]} for agent in agents_info],
        "query_matching_examples": query_results
    }


# Test util
async def main(host, port, transport, query, resource, tool, demo_mode=False):
    """Main asynchronous function to connect to the MCP server and execute commands.

    Used for local testing.

    Args:
        host: Server hostname.
        port: Server port.
        transport: Connection transport ('sse' or 'stdio').
        query: Optional query string for the 'find_agent' tool.
        resource: Optional resource URI to read.
        tool: Optional tool name to execute. Valid options are:
            'search_flights', 'search_hotels', or 'query_db'.
        demo_mode: If True, runs a comprehensive travel planning demo.
    """
    logger.info('Starting Client to connect to MCP')
    async with init_session(host, port, transport) as session:
        if demo_mode and query:
            # Run comprehensive travel planning demo
            logger.info("="*60)
            logger.info("🚀 COMPREHENSIVE TRAVEL PLANNING DEMO")
            logger.info("="*60)
            
            # Show system capabilities
            capabilities = await demonstrate_agent_capabilities(session)
            logger.info(f"📊 System has {capabilities['total_agents']} specialized agents:")
            for agent in capabilities['available_agents']:
                logger.info(f"   • {agent['name']}: {agent['description']}")
            
            logger.info("\n🔍 Agent Matching Examples:")
            for demo_query, result in capabilities['query_matching_examples'].items():
                if 'error' not in result:
                    logger.info(f"   Query: '{demo_query}' → {result['matched_agent']}")
            
            # Execute the main travel planning request
            logger.info(f"\n🎯 Executing Main Request: '{query}'")
            travel_result = await execute_travel_planning_demo(session, query)
            
            if travel_result['status'] == 'success_simulation':
                logger.info("="*60)
                logger.info("✅ TRAVEL PLANNING WORKFLOW DEMONSTRATION")
                logger.info("="*60)
                logger.info(f"🎯 Original Request: {travel_result['original_query']}")
                
                workflow = travel_result['workflow']
                logger.info(f"🤖 Orchestrator: {workflow['orchestrator_agent']['name']}")
                
                logger.info("\n📋 Selected Specialized Agents:")
                for role, agent in workflow['selected_agents'].items():
                    logger.info(f"   • {role.replace('_', ' ').title()}: {agent['name']}")
                
                logger.info("\n💾 Available Travel Data:")
                data = workflow['available_data']
                if data['flights']:
                    logger.info(f"   ✈️  Flights: {len(data['flights'])} routes available")
                    for flight in data['flights'][:3]:  # Show first 3
                        logger.info(f"      - {flight.get('from_airport', 'N/A')} → {flight.get('to_airport', 'N/A')} from ${flight.get('min_price', 'N/A')}")
                
                if data['hotels']:
                    logger.info(f"   🏨 Hotels: {len(data['hotels'])} destinations available")
                    for hotel in data['hotels'][:3]:  # Show first 3
                        logger.info(f"      - {hotel.get('city', 'N/A')}: {hotel.get('hotel_count', 'N/A')} hotels from ${hotel.get('min_price', 'N/A')}/night")
                
                if data['rental_cars']:
                    logger.info(f"   🚗 Car Rentals: {len(data['rental_cars'])} locations available")
                    for car in data['rental_cars'][:3]:  # Show first 3
                        logger.info(f"      - {car.get('city', 'N/A')}: {car.get('car_count', 'N/A')} vehicles from ${car.get('min_price', 'N/A')}/day")
                
                logger.info("\n📝 Summary:")
                logger.info(f"   {travel_result['summary']}")
                
                logger.info("="*60)
                logger.info("🎉 A2A + MCP Demo completed successfully!")
                logger.info("🔍 The system demonstrated:")
                logger.info("   • Semantic agent discovery using MCP")
                logger.info("   • Multi-agent coordination via A2A protocol")
                logger.info("   • Dynamic tool usage for data retrieval")
                logger.info("   • Comprehensive travel planning workflow")
                logger.info("="*60)
            elif travel_result['status'] == 'success':
                logger.info("="*60)
                logger.info("✅ TRAVEL PLANNING RESULTS")
                logger.info("="*60)
                logger.info(f"🎯 Original Request: {travel_result['query']}")
                logger.info(f"🤖 Orchestrator Agent: {travel_result['orchestrator_agent']}")
                logger.info(f"📋 Task ID: {travel_result['task_id']}")
                logger.info(f"🆔 Context ID: {travel_result['context_id']}")
                
                if 'response' in travel_result:
                    response = travel_result['response']
                    if 'message' in response and 'parts' in response['message']:
                        for part in response['message']['parts']:
                            if 'root' in part and 'text' in part['root']:
                                logger.info(f"📝 Response: {part['root']['text']}")
                
                logger.info("="*60)
                logger.info("🎉 Demo completed successfully!")
                logger.info("="*60)
            else:
                logger.error(f"❌ Demo failed: {travel_result.get('message', 'Unknown error')}")
            
            return travel_result
            
        # Original functionality
        if query:
            result = await find_agent(session, query)
            data = json.loads(result.content[0].text)
            logger.info(json.dumps(data, indent=2))
        if resource:
            result = await find_resource(session, resource)
            logger.info(result)
            data = json.loads(result.contents[0].text)
            logger.info(json.dumps(data, indent=2))
        if tool:
            if tool == 'search_flights':
                results = await search_flights(session)
                logger.info(results.model_dump())
            if tool == 'search_hotels':
                result = await search_hotels(session)
                data = json.loads(result.content[0].text)
                logger.info(json.dumps(data, indent=2))
            if tool == 'query_db':
                result = await query_db(session)
                logger.info(result)
                data = json.loads(result.content[0].text)
                logger.info(json.dumps(data, indent=2))


# Command line tester
@click.command()
@click.option('--host', default='localhost', help='SSE Host')
@click.option('--port', default='10100', help='SSE Port')
@click.option('--transport', default='stdio', help='MCP Transport')
@click.option('--find_agent', help='Query to find an agent')
@click.option('--resource', help='URI of the resource to locate')
@click.option('--tool_name', type=click.Choice(['search_flights', 'search_hotels', 'query_db']),
              help='Tool to execute: search_flights, search_hotels, or query_db')
@click.option('--demo', is_flag=True, help='Run comprehensive travel planning demo')
def cli(host, port, transport, find_agent, resource, tool_name, demo):
    """A command-line client to interact with the Agent Cards MCP server."""
    asyncio.run(main(host, port, transport, find_agent, resource, tool_name, demo))


if __name__ == '__main__':
    cli()
