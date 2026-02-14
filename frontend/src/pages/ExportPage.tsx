/**
 * Export page - placeholder for Camtrap DP export
 */

import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";

export default function ExportPage() {
  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Export</h1>
        <p className="text-muted-foreground mt-2">
          Export data in Camtrap DP format
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Camtrap DP Export</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12 space-y-4">
            <p className="text-lg text-muted-foreground">
              Coming soon
            </p>
            <div className="max-w-lg mx-auto text-sm text-muted-foreground space-y-2">
              <p>
                Export your project data as a Camera Trap Data Package (Camtrap DP),
                the community standard for sharing camera trap data.
              </p>
              <p>The export will include:</p>
              <ul className="text-left list-disc list-inside space-y-1">
                <li>
                  <strong>deployments.csv</strong> &mdash; camera deployment locations and dates
                </li>
                <li>
                  <strong>media.csv</strong> &mdash; image and video file metadata
                </li>
                <li>
                  <strong>observations.csv</strong> &mdash; species detections and classifications
                </li>
                <li>
                  <strong>datapackage.json</strong> &mdash; dataset descriptor with schema validation
                </li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
