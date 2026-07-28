/**
 * ExportScopeSelect — narrow a project export to specific sites /
 * deployments. Every deployment starts checked (= whole project); the
 * user unchecks to narrow. Picking a site toggles all its deployments.
 *
 * The picker owns the selection and reports a ready-to-send scope:
 *  - all deployments checked  → undefined (whole project, no URL filter)
 *  - a strict subset checked   → { deploymentIds: [...] }
 *  - nothing checked           → { deploymentIds: [] } (exports nothing)
 * Project mode only.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown } from "lucide-react";

import { sitesApi } from "../../api/sites";
import { deploymentsApi } from "../../api/deployments";
import type { ExportScope } from "../../api/export";
import type { DeploymentResponse } from "../../api/types";
import { basename } from "../../lib/path-utils";
import { Button } from "../ui/button";
import { Checkbox } from "../ui/checkbox";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";

interface Group {
  siteId: string | null;
  siteName: string;
  deployments: DeploymentResponse[];
}

interface ExportScopeSelectProps {
  projectId: string;
  /** Reports the resolved scope whenever the selection changes. */
  onChange: (scope: ExportScope | undefined) => void;
}

function deploymentLabel(d: DeploymentResponse): string {
  return basename(d.folder_path ?? "") || "(no folder)";
}

export function ExportScopeSelect({
  projectId,
  onChange,
}: ExportScopeSelectProps) {
  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
  });
  const { data: deployments } = useQuery({
    queryKey: ["deployments", projectId],
    queryFn: () => deploymentsApi.list({ projectId }),
  });

  const allIds = useMemo(
    () => (deployments ?? []).map((d) => d.id),
    [deployments],
  );
  const total = allIds.length;

  // Selected deployment ids. Starts empty and is filled with everything
  // once the deployment list first loads (so the default is whole
  // project). The ref guard keeps a later user "Clear" from being undone
  // by a refetch.
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const initialised = useRef(false);

  const scopeFor = (next: Set<string>): ExportScope | undefined =>
    next.size === total ? undefined : { deploymentIds: Array.from(next) };

  const apply = (next: Set<string>) => {
    setSelected(next);
    onChange(scopeFor(next));
  };

  useEffect(() => {
    if (!initialised.current && deployments) {
      initialised.current = true;
      const all = new Set(allIds);
      setSelected(all);
      onChange(scopeFor(all)); // all selected → undefined (whole project)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deployments]);

  // Group deployments by site: named sites first (alphabetical), then a
  // trailing "No site" group. Sites with no deployments are omitted.
  const groups = useMemo<Group[]>(() => {
    const nameById = new Map((sites ?? []).map((s) => [s.id, s.name]));
    const bySite = new Map<string | null, DeploymentResponse[]>();
    for (const d of deployments ?? []) {
      const list = bySite.get(d.site_id) ?? [];
      list.push(d);
      bySite.set(d.site_id, list);
    }
    const named: Group[] = [];
    let noSite: Group | null = null;
    for (const [siteId, deps] of bySite) {
      if (siteId === null) {
        noSite = { siteId: null, siteName: "No site", deployments: deps };
      } else {
        named.push({
          siteId,
          siteName: nameById.get(siteId) ?? "Unknown site",
          deployments: deps,
        });
      }
    }
    named.sort((a, b) => a.siteName.localeCompare(b.siteName));
    return noSite ? [...named, noSite] : named;
  }, [sites, deployments]);

  const toggleDeployment = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    apply(next);
  };

  const toggleSite = (group: Group) => {
    const ids = group.deployments.map((d) => d.id);
    const allSelected = ids.every((id) => selected.has(id));
    const next = new Set(selected);
    if (allSelected) ids.forEach((id) => next.delete(id));
    else ids.forEach((id) => next.add(id));
    apply(next);
  };

  const triggerLabel =
    total === 0 || selected.size === total
      ? "Whole project"
      : selected.size === 0
        ? "Nothing selected"
        : `${selected.size} of ${total} deployments`;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="justify-between gap-2 min-w-[200px]">
          <span className="truncate">{triggerLabel}</span>
          <ChevronDown className="h-4 w-4 shrink-0 opacity-60" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-72 p-0">
        <div className="flex items-center justify-between px-3 py-2 border-b">
          <span className="text-xs text-muted-foreground">
            {selected.size} of {total} selected
          </span>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-xs"
              disabled={selected.size === total}
              onClick={() => apply(new Set(allIds))}
            >
              All
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-xs"
              disabled={selected.size === 0}
              onClick={() => apply(new Set())}
            >
              Clear
            </Button>
          </div>
        </div>
        <div className="max-h-72 overflow-y-auto p-2">
          {groups.length === 0 ? (
            <p className="px-1 py-2 text-xs text-muted-foreground">
              No deployments to scope by.
            </p>
          ) : (
            groups.map((group) => {
              const ids = group.deployments.map((d) => d.id);
              const selectedCount = ids.filter((id) => selected.has(id)).length;
              const allSelected = selectedCount === ids.length;
              return (
                <div key={group.siteId ?? "__none__"} className="mb-1.5">
                  <label className="flex items-center gap-2 px-1 py-1 text-sm font-medium cursor-pointer">
                    <Checkbox
                      checked={allSelected}
                      indeterminate={selectedCount > 0 && !allSelected}
                      onCheckedChange={() => toggleSite(group)}
                    />
                    <span className="truncate">{group.siteName}</span>
                  </label>
                  <div className="pl-6">
                    {group.deployments.map((d) => (
                      <label
                        key={d.id}
                        className="flex items-center gap-2 px-1 py-0.5 text-xs text-muted-foreground cursor-pointer"
                      >
                        <Checkbox
                          checked={selected.has(d.id)}
                          onCheckedChange={() => toggleDeployment(d.id)}
                        />
                        <span className="truncate">{deploymentLabel(d)}</span>
                      </label>
                    ))}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
