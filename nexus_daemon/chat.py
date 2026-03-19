import requests
import json
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()
API_URL = "http://127.0.0.1:8000"
SESSION_ID = "local_nexus_session"

def main():
    console.print(Panel.fit("[bold cyan]NEXUS Asynchronous Local Assistant[/bold cyan]\nType '/quit' to exit, '/stats' for memory stats.", border_style="cyan"))
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
            if not user_input.strip():
                continue
                
            if user_input.lower() in ["/quit", "/exit"]:
                try:
                    requests.post(f"{API_URL}/save")
                except:
                    pass
                console.print("[bold cyan]NEXUS:[/bold cyan] Goodbye! Memories synced to disk.")
                break
                
            if user_input.lower() == "/stats":
                try:
                    res = requests.get(f"{API_URL}/stats")
                    if res.status_code == 200:
                        data = res.json()
                        console.print(f"[bold yellow]Storage:[/bold yellow] {data['storage_path']}")
                        console.print(f"[bold yellow]Episodes:[/bold yellow] {data['episodes_count']}")
                        console.print(f"[bold yellow]Palace Rooms:[/bold yellow] {data['palace_rooms']}")
                    else:
                        console.print(f"[red]Error fetching stats: {res.text}[/red]")
                except requests.exceptions.ConnectionError:
                    console.print("[red]Error: Make sure the daemon is running (uvicorn nexus_daemon.server:app --reload)[/red]")
                continue
                
            # Send Chat
            try:
                payload = {"session_id": SESSION_ID, "message": user_input}
                res = requests.post(f"{API_URL}/chat", json=payload)
                
                if res.status_code == 200:
                    data = res.json()
                    console.print("\n[bold cyan]NEXUS[/bold cyan]")
                    console.print(Markdown(data["response"]))
                else:
                    console.print(f"[red]Error {res.status_code}: {res.text}[/red]")
            except requests.exceptions.ConnectionError:
                console.print("[red]Error: Could not connect to NEXUS daemon. Start it with `python nexus_daemon/server.py`[/red]")
                
        except KeyboardInterrupt:
            try:
                requests.post(f"{API_URL}/save")
            except:
                pass
            console.print("\n[bold cyan]NEXUS:[/bold cyan] Goodbye! Memories synced to disk.")
            sys.exit(0)

if __name__ == "__main__":
    main()
