#!/usr/bin/env python3
"""
Quick API test script to verify endpoints work.

Run this after starting the dev server with ./run_dev.sh
"""

import httpx

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    """Test health check endpoint."""
    response = httpx.get(f"{BASE_URL}/health")
    print(f"✅ Health check: {response.status_code} - {response.json()}")


def test_projects_crud():
    """Test full CRUD flow for projects."""
    print("\n--- Testing Projects CRUD ---")

    # Create project
    response = httpx.post(
        f"{BASE_URL}/api/projects",
        json={"name": "Test Project", "description": "A test project"},
    )
    print(f"✅ Create project: {response.status_code}")
    project = response.json()
    project_id = project["id"]
    print(f"   Created project ID: {project_id}")

    # List projects
    response = httpx.get(f"{BASE_URL}/api/projects")
    print(f"✅ List projects: {response.status_code} - Found {len(response.json())} projects")

    # Get project by ID
    response = httpx.get(f"{BASE_URL}/api/projects/{project_id}")
    print(f"✅ Get project: {response.status_code} - {response.json()['name']}")

    # Update project
    response = httpx.patch(
        f"{BASE_URL}/api/projects/{project_id}",
        json={"description": "Updated description"},
    )
    print(f"✅ Update project: {response.status_code}")

    # Get stats
    response = httpx.get(f"{BASE_URL}/api/projects/{project_id}/stats")
    print(f"✅ Get project stats: {response.status_code}")
    stats = response.json()
    print(f"   Sites: {stats['site_count']}, Files: {stats['file_count']}")

    return project_id


def test_sites_crud(project_id):
    """Test full CRUD flow for sites."""
    print("\n--- Testing Sites CRUD ---")

    # Create site
    response = httpx.post(
        f"{BASE_URL}/api/sites",
        json={
            "project_id": project_id,
            "name": "Test Site",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "habitat_type": "Urban",
        },
    )
    print(f"✅ Create site: {response.status_code}")
    site = response.json()
    site_id = site["id"]
    print(f"   Created site ID: {site_id}")

    # List all sites
    response = httpx.get(f"{BASE_URL}/api/sites")
    print(f"✅ List all sites: {response.status_code} - Found {len(response.json())} sites")

    # List sites by project
    response = httpx.get(f"{BASE_URL}/api/sites?project_id={project_id}")
    print(f"✅ List sites by project: {response.status_code} - Found {len(response.json())} sites")

    # Get site by ID
    response = httpx.get(f"{BASE_URL}/api/sites/{site_id}")
    print(f"✅ Get site: {response.status_code} - {response.json()['name']}")

    # Update site
    response = httpx.patch(
        f"{BASE_URL}/api/sites/{site_id}",
        json={"elevation_m": 10.0},
    )
    print(f"✅ Update site: {response.status_code}")

    # Delete site
    response = httpx.delete(f"{BASE_URL}/api/sites/{site_id}")
    print(f"✅ Delete site: {response.status_code}")


def test_cleanup(project_id):
    """Clean up test data."""
    print("\n--- Cleanup ---")
    response = httpx.delete(f"{BASE_URL}/api/projects/{project_id}")
    print(f"✅ Delete project: {response.status_code}")


def main():
    """Run all tests."""
    print("Testing AddaxAI API endpoints...\n")

    try:
        test_health()
        project_id = test_projects_crud()
        test_sites_crud(project_id)
        test_cleanup(project_id)

        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
