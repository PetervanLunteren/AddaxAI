/**
 * TypeScript types for API requests and responses.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Matches backend Pydantic schemas
 */

// Project types
export interface ProjectCreate {
  name: string;
  description?: string | null;
}

export interface ProjectUpdate {
  name?: string | null;
  description?: string | null;
}

export interface ProjectResponse {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProjectWithStats extends ProjectResponse {
  site_count: number;
  deployment_count: number;
  file_count: number;
  detection_count: number;
}

// Site types
export interface SiteCreate {
  project_id: string;
  name: string;
  latitude?: number | null;
  longitude?: number | null;
  elevation_m?: number | null;
  habitat_type?: string | null;
  notes?: string | null;
}

export interface SiteUpdate {
  name?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  elevation_m?: number | null;
  habitat_type?: string | null;
  notes?: string | null;
}

export interface SiteResponse {
  id: string;
  project_id: string;
  name: string;
  latitude: number | null;
  longitude: number | null;
  elevation_m: number | null;
  habitat_type: string | null;
  notes: string | null;
  created_at: string;
}
