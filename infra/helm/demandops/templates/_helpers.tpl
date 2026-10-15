{{- define "demandops.labels" -}}
app.kubernetes.io/name: demandops
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{- define "demandops.image" -}}
{{- $root := index . 0 -}}
{{- $name := index . 1 -}}
{{ $root.Values.image.registry }}/{{ $name }}:{{ $root.Values.image.tag }}
{{- end }}

{{/* Env shared by every pod: plain values + the credentials secret */}}
{{- define "demandops.env" -}}
{{- range $k, $v := .Values.env }}
- name: {{ $k }}
  value: {{ $v | quote }}
{{- end }}
{{- range $k := list "AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY" "DATABASE_URL" }}
- name: {{ $k }}
  valueFrom:
    secretKeyRef:
      name: {{ $.Values.existingSecret }}
      key: {{ $k }}
{{- end }}
{{- end }}

{{/* Env specific to the serving pods */}}
{{- define "demandops.serving.env" -}}
- name: MODEL_VERSION
  value: {{ .Values.model.version | quote }}
- name: MODEL_NAME
  value: {{ .Values.model.name | quote }}
- name: FEATURE_BACKEND
  value: {{ .Values.serving.featureBackend | quote }}
{{- if .Values.serving.firstTargetDay }}
- name: FIRST_TARGET_DAY
  value: {{ .Values.serving.firstTargetDay | quote }}
{{- end }}
{{- end }}
